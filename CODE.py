import os, json, re
from collections import defaultdict
import heapq
import argparse
import hashlib
from bs4 import BeautifulSoup

TOKENS = re.compile(r"[A-Za-z0-9]+")

class Stemmer:
    def stem(self, w: str) -> str:
        w = w.lower()
        if len(w) <= 3:
            return w
        if w.endswith("ing") and len(w) > 5:
            return w[:-3]
        if w.endswith("ied") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ed") and len(w) > 4:
            return w[:-2]
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 3:
            return w[:-2]
        if w.endswith("s") and len(w) > 3:
            return w[:-1]
        return w

def extract_texts_bs4(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    important_chunks = []

    for tag in soup.find_all(["title", "h1", "h2", "h3", "b", "strong"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            important_chunks.append(txt)

    body_text = soup.get_text(" ", strip=True)

    return " ".join(important_chunks), body_text

class Indexer:
    def __init__(self, corpus_root, out_dir, flush_threshold=200_000):
        self.corpus_root = corpus_root
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.stemmer = Stemmer()
        self.index = defaultdict(dict)
        self.doc_id = 0
        self.urls = []
        self.flush_threshold = flush_threshold
        self.partial_count = 0

        self.doc_lengths = {}
        self.simhash_buckets = defaultdict(list)

        self.IMP_WEIGHT = 3.0
        self.BODY_WEIGHT = 1.0
        self.SIMHASH_BITS = 64
        self.THRESHHOLD = 3

    def tokens(self, text):
        for m in TOKENS.finditer(text.lower()):
            yield self.stemmer.stem(m.group(0))

    def process_document(self, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        html = obj.get("content", "")
        url = obj.get("url", "")

        if isinstance(html, str):
            s = html.strip()

            if s.startswith("http://") or s.startswith("https://"):
                return

        low_html = html.lower()

        if ("the requested url was not found on this server" in low_html or "<title>not found</title>" in low_html):
            return

        #MISSING OR NOT CONTENT
        if not html or len(html.strip()) < 50:
            return
        
        low = html.lstrip().lower()
        if low.startswith("{") or low.startswith("["):
            return

        try:
            imp_text, body_text = extract_texts_bs4(html)
        except Exception:
            return

        combined = (imp_text + " " + body_text).strip()

        combined_low = combined.lower()

        if ("the requested url was not found on this server" in combined_low or combined_low.startswith("not found") and "apache" in combined_low):
            return

        if len(combined) < 50:
            return

        tf_imp, tf_other = defaultdict(int), defaultdict(int)

        for t in self.tokens(imp_text):
            tf_imp[t] += 1
        for t in self.tokens(body_text):
            tf_other[t] += 1

        #Too little tokens
        if sum(tf_imp.values()) + sum(tf_other.values()) < 5:
            return

        #FOR PERFECT AND NEAR DUPLICATES
        sh = self.simhash(tf_imp, tf_other)
        if self.is_near_duplicate(sh):
            return

        self.doc_id += 1
        doc_id = self.doc_id
        self.urls.append(url)

        self.remember_simhash(sh, doc_id)

        imp_count = sum(tf_imp.values())
        body_count = sum(tf_other.values())
        raw_len_score = self.IMP_WEIGHT * imp_count + self.BODY_WEIGHT * body_count

        if raw_len_score < 10:
            return
        
        self.doc_lengths[doc_id] = raw_len_score

        for term in set(tf_imp.keys()) | set(tf_other.keys()):
            self.index[term][doc_id] = [tf_imp.get(term, 0), tf_other.get(term, 0)]

        if self._need_flush():
            self.flush_partial()

    def _need_flush(self):

        return sum(len(docs) for docs in self.index.values()) >= self.flush_threshold

    def flush_partial(self):
        # Flush to not run out of memory
        if not self.index:
            return
        self.partial_count += 1
        fname = os.path.join(self.out_dir, f"partial_{self.partial_count}.jsonl")
        print(f"flush -> {fname}")

        with open(fname, "w", encoding="utf-8") as f:
            for term in sorted(self.index.keys()):
                postings = [
                    [doc_id, vals[0], vals[1]]
                    for doc_id, vals in sorted(self.index[term].items())
                ]
                f.write(json.dumps({"term": term, "postings": postings}) + "\n")

        self.index.clear()

    def walk_corpus(self):
        for root, _, files in os.walk(self.corpus_root):
            for name in files:
                if name.endswith(".json"):
                    self.process_document(os.path.join(root, name))

        self.flush_partial()

    def simhash(self, tf_imp, tf_other):
        weights = defaultdict(int)

        for t, c in tf_imp.items():
            weights[t] += 3 * c

        for t, c in tf_other.items():
            weights[t] += 1 * c

        v = [0] * self.SIMHASH_BITS

        for term, w in weights.items():

            h = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16)

            for i in range(self.SIMHASH_BITS):
                bit = (h >> i) & 1
                v[i] += w if bit else -w

        out = 0
        for i in range(self.SIMHASH_BITS):
            if v[i] > 0:
                out |= (1 << i)
        return out

    def hamming(self, a, b):
        return (a ^ b).bit_count()

    def simhash_bucket_key(self, sh):
        return sh >> (self.SIMHASH_BITS - 16)

    def is_near_duplicate(self, sh):
        key = self.simhash_bucket_key(sh)

        for prev_sh, _prev_doc in self.simhash_buckets[key]:
            if self.hamming(sh, prev_sh) <= self.THRESHHOLD:
                return True
            
        return False

    def remember_simhash(self, sh, doc_id):
        key = self.simhash_bucket_key(sh)
        self.simhash_buckets[key].append((sh, doc_id))

    def open_partial_iter(self, filepath):
        f = open(filepath, "r", encoding="utf-8")
        line = f.readline()

        if not line:
            f.close()
            return None
        
        obj = json.loads(line)

        return {"file": f, "term": obj["term"], "postings": obj["postings"]}

    def write_doc_index(self):
        doc_path = os.path.join(self.out_dir, "doc_index.jsonl")
        with open(doc_path, 'w', encoding="utf-8") as f:
            for i, url in enumerate(self.urls, start=1):
                f.write(json.dumps({"doc_id": i, "url": url}) + "\n")

        meta_path = os.path.join(self.out_dir, "meta.json")
        with open(meta_path, 'w', encoding="utf-8") as f:
            json.dump({"num_docs": self.doc_id}, f)

        stats_path = os.path.join(self.out_dir, "stats.json")
        with open(stats_path, 'w', encoding="utf-8") as f:
            json.dump({"doc_lengths": self.doc_lengths}, f)

    def merge_partials(self):
        partial_files = sorted(
            f for f in os.listdir(self.out_dir)
            if f.startswith("partial_") and f.endswith(".jsonl")
        )
        if not partial_files:
            print("No partial files to merge.")
            return

        final_path = os.path.join(self.out_dir, "merged_index.jsonl")
        lexicon_path = os.path.join(self.out_dir, "lexicon.json")

        heap = []
        streams = []

        for pf in partial_files:
            st = self.open_partial_iter(os.path.join(self.out_dir, pf))
            if st:
                streams.append(st)
                heapq.heappush(heap, (st["term"], len(streams) - 1))

        lexicon = {}

        with open(final_path, "w", encoding="utf-8") as out:
            while heap:
                term, idx = heapq.heappop(heap)

                same = [idx]
                while heap and heap[0][0] == term:
                    _, idx2 = heapq.heappop(heap)
                    same.append(idx2)

                merged = defaultdict(lambda: [0, 0])

                for si in same:
                    postings = streams[si]["postings"]
                    for doc_id, tf_imp, tf_body in postings:
                        doc_id = int(doc_id)
                        merged[doc_id][0] += int(tf_imp)
                        merged[doc_id][1] += int(tf_body)

                merged_postings = [
                    [doc_id, vals[0], vals[1]]
                    for doc_id, vals in sorted(merged.items())
                ]

                offset = out.tell()
                out.write(json.dumps({"term": term, "postings": merged_postings}) + "\n")
                lexicon[term] = {"offset": offset, "df": len(merged_postings)}

                for si in same:
                    f = streams[si]["file"]
                    line = f.readline()
                    if line:
                        obj = json.loads(line)
                        streams[si]["term"] = obj["term"]
                        streams[si]["postings"] = obj["postings"]
                        heapq.heappush(heap, (streams[si]["term"], si))
                    else:
                        f.close()

        with open(lexicon_path, "w", encoding="utf-8") as f:
            json.dump(lexicon, f)

        print(f"Merged -> {final_path}")
        print(f"Lexicon -> {lexicon_path}")

    def build(self):
        self.walk_corpus()
        self.write_doc_index()
        self.merge_partials()
        print("Indexing and merging complete")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--flush_threshold", type=int, default=500_000)
    args = ap.parse_args()

    idx = Indexer(args.corpus_root, args.data, args.flush_threshold)
    idx.build()