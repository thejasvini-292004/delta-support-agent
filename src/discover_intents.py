"""
Step 1 — Discover an intent taxonomy from the data (unsupervised).

We do NOT invent intents from our heads. We cluster the real customer messages,
read the top terms + representative examples of each cluster, and use that to
NAME a small taxonomy by hand (written up in intents_proposed.md).

Method: TF-IDF (1-2 grams) -> KMeans. Over-segment (k=12), then merge into 6-8.

Usage: python src/discover_intents.py --k 12
"""
import argparse, os, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

STOP_EXTRA = {"delta","amp","im","ive","dont","cant","get","got","just","like",
              "please","pls","thanks","thank","hi","hello","hey","u","ur","w"}

def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here,"..","data","delta_customer_messages.csv"))
    ap.add_argument("--out",  default=os.path.join(here,"..","data","clusters_report.txt"))
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    texts = df["clean"].fillna("").astype(str)
    texts = texts[texts.str.len() >= 5].reset_index(drop=True)
    print(f"clustering {len(texts):,} customer messages, k={a.k}")

    from sklearn.feature_extraction import text as sktext
    stop = list(sktext.ENGLISH_STOP_WORDS.union(STOP_EXTRA))
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1,2),
                          min_df=10, max_df=0.4, max_features=20000)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=a.k, random_state=a.seed, n_init=5).fit(X)
    labels = km.labels_
    terms = np.array(vec.get_feature_names_out())
    centroids = km.cluster_centers_

    lines = []
    order = np.argsort(-np.bincount(labels, minlength=a.k))  # biggest clusters first
    for c in order:
        idx = np.where(labels == c)[0]
        size = len(idx); pct = 100*size/len(texts)
        top = terms[np.argsort(-centroids[c])[:12]]
        # representative examples = closest to centroid
        sims = np.asarray(X[idx] @ centroids[c].reshape(-1,1)).ravel()
        reps = idx[np.argsort(-sims)[:5]]
        lines.append(f"\n=== cluster {c}  |  {size:,} msgs ({pct:.1f}%) ===")
        lines.append("top terms: " + ", ".join(top))
        for r in reps:
            lines.append("   • " + texts.iloc[r][:130])
    report = "\n".join(lines)
    with open(a.out,"w") as f: f.write(report)
    print(report)

if __name__ == "__main__":
    main()
