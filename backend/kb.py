"""Knowledge base: BM25 keyword retrieval over the ingested public documents.

No embedding model: the server runs on a 512 MB instance, and BM25 is transparent
and fast. Domain vocabulary gaps (a query says "seagrass", the documents say
"eelgrass") are handled by an explicit thesaurus, and the retrieval test measures
how much that thesaurus helps.
"""
import json, math, pathlib, re
from collections import Counter

KB_DIR = pathlib.Path(__file__).resolve().parent / "kb"

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this
to was were will with what which how why who when where does do did can could should would
about into than then there their these those not no but if so such""".split())

# Domain synonyms: each group is expanded both ways at query time.
THESAURUS = [
    {"eelgrass", "seagrass", "zostera"},
    {"bull", "nereocystis"},
    {"heatwave", "heatwaves", "heat"},
    {"orca", "orcas", "whale", "whales"},
    {"salmon", "chinook"},
]

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9,.-]*[a-z0-9]|[a-z0-9]", text.lower())
    return [w.replace(",", "") for w in words if w not in STOPWORDS]

def expand(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        for group in THESAURUS:
            if t in group:
                out.extend(g for g in group if g != t)
    return out


class KnowledgeBase:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.chunks = json.loads((KB_DIR / "chunks.json").read_text())
        self.report = json.loads((KB_DIR / "ingest_report.json").read_text())
        self.k1, self.b = k1, b
        self.docs_tf = [Counter(tokenize(c["text"])) for c in self.chunks]
        self.doc_len = [sum(tf.values()) for tf in self.docs_tf]
        self.avg_len = sum(self.doc_len) / len(self.doc_len)
        df = Counter(t for tf in self.docs_tf for t in tf)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4, use_thesaurus: bool = True) -> list[dict]:
        terms = tokenize(query)
        if use_thesaurus:
            terms = expand(terms)
        scores = []
        for i, tf in enumerate(self.docs_tf):
            s = 0.0
            for t in set(terms):
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [{**{k_: self.chunks[i][k_] for k_ in ("id", "doc_id", "title", "publisher", "year", "url", "page", "text")},
                 "score": round(s, 3)} for s, i in scores[:k]]


EVAL_SET = [
    # (question, regex that the evidence passage must contain)
    ("How many acres of kelp and eelgrass does Washington aim to conserve by 2040?", r"10,000 acres"),
    ("How many goals and actions does the Puget Sound Kelp Plan contain?", r"65 actions|six (strategic )?goals"),
    ("What is the geographic scope of the Kelp Plan?", r"Georgia Strait"),
    ("How do overwater structures affect kelp?", r"overwater structure"),
    ("What stressors are driving bull kelp declines in Puget Sound?", r"warming ocean temperatures"),
    ("How have marine heat waves affected kelp forests?", r"marine heat ?waves?"),
    ("Which kelp species forms the floating canopy in Puget Sound?", r"Nereocystis luetkeana"),
    ("What is the Kelp Forest Monitoring Alliance?", r"Monitoring Alliance"),
    ("What protection zone did DNR create in Snohomish County?", r"2,300-acre"),
    # Vocabulary-gap questions: phrased with words the documents rarely use.
    ("How have heatwaves hurt bull kelp?", r"heat ?waves?"),
    ("Why is seagrass habitat important for young salmon?", r"eelgrass"),
    ("Where has Nereocystis canopy declined?", r"declin"),
]

def run_eval(kb: KnowledgeBase, k: int = 4) -> dict:
    def score(use_thesaurus: bool):
        rows, hits, rr = [], 0, 0.0
        for q, pattern in EVAL_SET:
            results = kb.search(q, k=k, use_thesaurus=use_thesaurus)
            rank = next((i + 1 for i, r in enumerate(results) if re.search(pattern, r["text"], re.I)), None)
            hits += rank is not None
            rr += 1 / rank if rank else 0
            rows.append({"question": q, "evidence": pattern, "found_at_rank": rank,
                         "top_source": f"{results[0]['title']}, p.{results[0]['page']}" if results else None})
        n = len(EVAL_SET)
        return {"hit_rate": round(hits / n, 3), "mrr": round(rr / n, 3), "hits": hits, "total": n, "questions": rows}
    return {"k": k, "method": "BM25 keyword retrieval",
            "with_thesaurus": score(True), "without_thesaurus": score(False)}
