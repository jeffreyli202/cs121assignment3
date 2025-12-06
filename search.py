import os
import json
import re
import math
import argparse
from collections import defaultdict

TOKENS = re.compile(r"[A-Za-z0-9]+")

class Stemmer:
    def stem(self, w):
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


class Searcher:
    IMP_WEIGHT = 3.0
    BODY_WEIGHT = 1.0

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.stemmer = Stemmer()

        self.doc_index = self.load_doc_index()
        self.num_docs = self.load_num_docs()
        self.lexicon = self.load_lexicon()
        self.doc_lengths = self.load_doc_lengths()

        self.final_index_path = os.path.join(self.data_dir, "merged_index.jsonl")
        if not os.path.exists(self.final_index_path):
            raise FileNotFoundError("merged_index.jsonl not found. Merger not working?")

    def load_doc_index(self):
        path = os.path.join(self.data_dir, "doc_index.jsonl")

        if not os.path.exists(path):
            return {}
        
        doc_index = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = int(obj["doc_id"])
                url = obj["url"]
                doc_index[doc_id] = url

        return doc_index

    def load_num_docs(self):
        path = os.path.join(self.data_dir, "meta.json")

        if not os.path.exists(path):
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return int(obj["num_docs"])
    
    def load_lexicon(self):
        path = os.path.join(self.data_dir, "lexicon.json")

        if not os.path.exists(path):
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def load_doc_lengths(self):
        path = os.path.join(self.data_dir, "stats.json")

        if not os.path.exists(path):
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return {int(k): float(v) for k, v in obj.get("doc_lengths", {}).items()}

    def tokens(self, text):
        for m in TOKENS.finditer(text.lower()):
            yield self.stemmer.stem(m.group(0))

    def url_tokens(self, url):
        parts = re.findall(r"[A-Za-z0-9]+", url.lower())
        return set(self.stemmer.stem(p) for p in parts if p)
    
    def major_heuristic(self, scores, query_terms):
        qset = set(query_terms)

        for doc_id in list(scores.keys()):
            url = self.doc_index.get(doc_id, "")
            u = url.lower()

            mult = 1.0

            if "/category/" in u or "/tag/" in u or "/author/" in u:
                mult *= 0.60

            if "/~" in u:
                mult *= 0.85

            if "major" in qset or "requirements" in qset or "degree" in qset:
                if "/ugrad/" in u or "undergrad" in u:
                    mult *= 1.20
                if "change_of_major" in u:
                    mult *= 0.80

            scores[doc_id] *= mult
        
    def coverage_heuristic(self, scores, term_postings, query_terms):
        qn = len(query_terms)
        if qn == 0:
            return

        for doc_id in list(scores.keys()):
            matched = 0
            for t in query_terms:
                p = term_postings.get(t)
                if p and doc_id in p:
                    matched += 1

            coverage = matched / qn
            scores[doc_id] *= (1.0 + 0.40 * coverage)

    def get_postings_for_term(self, term):
        info = self.lexicon.get(term)

        if not info:
            return {}

        offset = info["offset"]

        with open(self.final_index_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            if not line:
                return {}
            obj = json.loads(line)
            if obj.get("term") != term:
                return {}

        postings = {}
        for doc_id, tf_imp, tf_body in obj["postings"]:
            postings[int(doc_id)] = (int(tf_imp), int(tf_body))

        return postings

    def search(self, query_str, top_k=10):
        terms = [t for t in self.tokens(query_str)]
        if not terms:
            return []

        seen = set()
        unique_terms = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)

        term_postings = {}
        for t in unique_terms:
            p = self.get_postings_for_term(t)
            if p:
                term_postings[t] = p
        
        if not term_postings:
            return []

        candidate_docs = set()
        for p in term_postings.values():
            candidate_docs |= set(p.keys())

        scores = defaultdict(float)
        N = self.num_docs

        for t, p in term_postings.items():
            df = len(p)
            idf = math.log((N + 1.0) / (df + 1.0)) + 1.0

            for doc_id in candidate_docs:
                if doc_id not in p:
                    continue
                tf_imp, tf_body = p[doc_id]
                tf = self.IMP_WEIGHT * tf_imp + self.BODY_WEIGHT * tf_body
                if tf > 0:
                    scores[doc_id] += tf * idf

        for doc_id in list(scores.keys()):
            dl = self.doc_lengths.get(doc_id)
            if dl and dl > 0:
                scores[doc_id] /= math.sqrt(dl)

        self.coverage_heuristic(scores, term_postings, unique_terms)
        self.major_heuristic(scores, unique_terms)

        ranked = sorted(
            ((score, doc_id) for doc_id, score in scores.items()),
            key=lambda x: (-x[0], x[1])
        )
        return ranked[:top_k]

    def print_results(self, query_str, top_k=1):
        print(f"\nQuery: {query_str!r}")
        results = self.search(query_str, top_k=top_k)
        if not results:
            print("Cannot find any results.")
            return

        for rank, (s, doc_id) in enumerate(results, start=1):
            url = self.doc_index.get(doc_id, "<unknown url>")
            print(f'{rank}: {url}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    searcher = Searcher(args.data)

    while True:
        try:
            q = input("query:  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            break

        searcher.print_results(q, top_k=args.topk)


if __name__ == "__main__":
    main()
