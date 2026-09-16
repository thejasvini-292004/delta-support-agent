"""Grounding retriever: TF-IDF over historical (customer_msg -> brand_reply) pairs.
Only 'groundable' pairs (real resolutions, not DM-deflections) are indexed."""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, pairs_path=None):
        here = os.path.dirname(os.path.abspath(__file__))
        pairs_path = pairs_path or os.path.join(here,"..","data","delta_grounding_pairs.jsonl")
        self.msgs, self.replies = [], []
        for line in open(pairs_path):
            p = json.loads(line)
            if p.get("groundable") and p.get("customer_msg"):
                self.msgs.append(p["customer_msg"]); self.replies.append(p["brand_reply"])
        self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=40000)
        self.M = self.vec.fit_transform(self.msgs)

    def retrieve(self, query:str, k=3):
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self.M).ravel()
        idx = np.argsort(-sims)[:k]
        return [{"customer_msg":self.msgs[i], "brand_reply":self.replies[i],
                 "score":float(sims[i])} for i in idx]
