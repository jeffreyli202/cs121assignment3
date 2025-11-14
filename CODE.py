import os, json, re
from collections import defaultdict
from html.parser import HTMLParser
import argparse

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

class HTML(HTMLParser):

    TAGS = {"title", "h1", "h2", "h3", "b", "strong"}

    def __init__(self):
        super().__init__()
        self.in_important = 0
        self.imp_chunks = []
        self.body_chunks = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.TAGS:
            self.in_important += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.TAGS and self.in_important > 0:
            self.in_important -= 1

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        self.body_chunks.append(text)
        if self.in_important:
            self.imp_chunks.append(text)

    def get_texts(self):
        return " ".join(self.imp_chunks), " ".join(self.body_chunks)


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

    def tokens(self, text):
        for m in TOKENS.finditer(text.lower()):
            yield self.stemmer.stem(m.group(0))

    def process_document(self, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        html = obj.get("content", "")
        url = obj.get("url", "")

        self.doc_id += 1
        doc_id = self.doc_id
        self.urls.append(url)

        parser = HTML()
        parser.feed(html)
        imp_text, body_text = parser.get_texts()

        tf_imp, tf_other = defaultdict(int), defaultdict(int)

        for t in self.tokens(imp_text):
            tf_imp[t] += 1
        for t in self.tokens(body_text):
            tf_other[t] += 1

        for term in set(tf_imp.keys()) | set(tf_other.keys()):
            self.index[term][doc_id] = [tf_imp.get(term, 0), tf_other.get(term, 0)]

        if self._need_flush():
            self.flush_partial()

    def _need_flush(self):

        return sum(len(docs) for docs in self.index.values()) >= self.flush_threshold

    def flush_partial(self):
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

    def write_doc_index(self):
        doc_path = os.path.join(self.out_dir, "doc_index.jsonl")
        with open(doc_path, "w", encoding="utf-8") as f:
            for i, url in enumerate(self.urls, start=1):
                f.write(json.dumps({"doc_id": i, "url": url}) + "\n")
        meta_path = os.path.join(self.out_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"num_docs": self.doc_id}, f)

    def build(self):
        self.walk_corpus()
        self.write_doc_index()
        print("Indexing complete on partials.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--flush_threshold", type=int, default=500_000)
    args = ap.parse_args()

    idx = Indexer(args.corpus_root, args.data, args.flush_threshold)
    idx.build()