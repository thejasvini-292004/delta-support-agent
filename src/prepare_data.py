"""
Step 0 — Reproducible data prep for the Hiver assignment.

Reads the raw Twitter customer-support CSV, filters to one brand (default: Delta),
cleans the text, reconstructs conversation threads via the reply links, and writes:

  data/delta_grounding_pairs.jsonl   (customer_msg -> brand_reply) pairs = the RAG / grounding corpus
  data/delta_customer_messages.csv   deduped inbound customer messages to the brand = intent dataset
  data/delta_sample.csv              a small, seeded random sample for fast iteration

The raw file is never modified. Everything is deterministic given --seed.

Usage:
  python src/prepare_data.py                       # defaults: Delta, ../twcs/twcs.csv
  python src/prepare_data.py --brand AmazonHelp --sample 500
"""
import argparse, json, os, re, sys
import pandas as pd

# ---- text cleaning -------------------------------------------------------
URL_RE  = re.compile(r"https?://\S+")
MENT_RE = re.compile(r"@\w+")
SIG_RE  = re.compile(r"[\*\^~][A-Z]{1,4}\s*$")      # agent signature, e.g. "*TJH", "^AB"
PART_RE = re.compile(r"\b\d+/\d+\s*$")               # multi-part marker, e.g. "1/2"
WS_RE   = re.compile(r"\s+")

def clean(text: str) -> str:
    """Light, reversible-in-spirit cleaning. We KEEP a raw copy elsewhere."""
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub(" ", text)      # drop t.co links
    t = SIG_RE.sub("", t)          # drop trailing agent initials
    t = PART_RE.sub("", t)         # drop "1/2" style markers
    t = MENT_RE.sub(" ", t)        # drop @handles (incl. anonymized @12345)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = WS_RE.sub(" ", t).strip()
    return t

def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.environ.get("TWCS_CSV",
                    os.path.join(here, "..", "..", "twcs", "twcs.csv")))
    ap.add_argument("--brand", default="Delta")
    ap.add_argument("--out",   default=os.path.join(here, "..", "data"))
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--chunk",  type=int, default=500_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    brand_re = re.compile(rf"@?{re.escape(a.brand)}\b", re.I)

    # ---- Pass A: collect this brand's outbound replies + the parent ids they answer
    replies = []            # (reply_id, parent_id, reply_raw, reply_clean, created_at)
    parent_ids = set()
    cols = ["tweet_id","author_id","inbound","created_at","text","in_response_to_tweet_id"]
    for ch in pd.read_csv(a.csv, usecols=cols, chunksize=a.chunk, dtype=str):
        d = ch[(ch["author_id"] == a.brand) & (ch["inbound"] == "False")]
        for tid, pid, raw, ts in zip(d["tweet_id"], d["in_response_to_tweet_id"],
                                     d["text"].fillna(""), d["created_at"]):
            replies.append((tid, pid, raw, clean(raw), ts))
            if isinstance(pid, str):
                parent_ids.add(pid)
    print(f"[A] {a.brand} outbound replies: {len(replies):,} | parent ids to fetch: {len(parent_ids):,}")

    # ---- Pass B: fetch parent (customer) texts + all inbound customer msgs to the brand
    id2raw = {}
    cust_rows = []          # inbound customer messages that mention the brand
    for ch in pd.read_csv(a.csv, usecols=cols, chunksize=a.chunk, dtype=str):
        m = ch[ch["tweet_id"].isin(parent_ids)]
        id2raw.update(zip(m["tweet_id"], m["text"].fillna("")))
        c = ch[(ch["inbound"] == "True") & (ch["text"].fillna("").str.contains(brand_re))]
        for tid, raw, ts, pid in zip(c["tweet_id"], c["text"].fillna(""),
                                     c["created_at"], c["in_response_to_tweet_id"]):
            cust_rows.append((tid, raw, clean(raw), ts, pid))
    print(f"[B] parent texts found: {len(id2raw):,} | inbound customer msgs: {len(cust_rows):,}")

    # ---- Grounding pairs: reply joined to the customer message it answered
    pairs, dm_deflect = [], 0
    DM_RE = re.compile(r"\b(dm|direct message|private message|inbox|send us a)\b", re.I)
    for rid, pid, r_raw, r_clean, ts in replies:
        cust_raw = id2raw.get(pid)
        if not cust_raw:                 # parent not a customer tweet / not found
            continue
        is_dm = bool(DM_RE.search(r_raw))
        dm_deflect += int(is_dm)
        pairs.append({
            "reply_id": rid, "parent_id": pid,
            "customer_msg": clean(cust_raw), "customer_raw": cust_raw,
            "brand_reply": r_clean, "brand_reply_raw": r_raw,
            "created_at": ts,
            "is_dm_deflection": is_dm,          # exclude these from grounding, keep as signal
            "groundable": (not is_dm) and len(r_clean) > 15,
        })
    with open(os.path.join(a.out, "delta_grounding_pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    groundable = sum(p["groundable"] for p in pairs)
    print(f"[join] grounding pairs: {len(pairs):,} | DM-deflection: {dm_deflect:,} "
          f"({100*dm_deflect/max(len(pairs),1):.1f}%) | groundable: {groundable:,}")

    # ---- Intent dataset: dedup customer messages (drop exact-dup cleaned text = viral tweets)
    cust = pd.DataFrame(cust_rows, columns=["tweet_id","raw","clean","created_at","in_response_to"])
    cust = cust[cust["clean"].str.len() >= 5].copy()
    before = len(cust)
    cust = cust.drop_duplicates(subset="clean").reset_index(drop=True)
    print(f"[intent] customer msgs: {before:,} -> {len(cust):,} after exact-dedup (viral tweets)")
    cust.to_csv(os.path.join(a.out, "delta_customer_messages.csv"), index=False)

    # ---- Seeded sample for fast iteration
    n = min(a.sample, len(cust))
    cust.sample(n=n, random_state=a.seed).to_csv(
        os.path.join(a.out, "delta_sample.csv"), index=False)
    print(f"[sample] wrote {n} rows (seed={a.seed}) -> data/delta_sample.csv")
    print("DONE.")

if __name__ == "__main__":
    main()
