"""
Step 2 (setup) — Build a STRATIFIED candidate pool for you to hand-label.

We do NOT label for you (the assignment wants your golden set, and you'll defend it).
Instead we:
  1. Weak-label every message with a transparent keyword rule into the proposed intents,
     so we can (a) estimate class sizes and (b) make sure rare intents get sampled.
  2. Draw a stratified sample (floor per class) + always include some questions and
     non-actionable noise, and write a template with BLANK columns for you to fill.

The 'suggested_intent' is only a hint to speed your labelling — correct it freely.

Output: data/golden_to_label.csv  (columns: tweet_id, text, suggested_intent,
        intent[blank], decision[blank: auto|escalate], notes[blank])
"""
import argparse, os, re
import pandas as pd

# --- Proposed taxonomy as transparent keyword rules (priority = list order) ---
RULES = [
 ("flight_disruption", r"\b(delay|delayed|cancel|cancell?ed|cancellation|missed? connection|diverted|stranded|stuck|rebook)\b"),
 ("baggage",           r"\b(bag|bags|baggage|luggage|suitcase|carry.?on|checked bag|gate check|lost bag)\b"),
 ("booking_change",    r"\b(change my flight|reschedul|standby|basic economy|refund|change fee|itinerary|cancel my|book(ing)?)\b"),
 ("seating_upgrade",   r"\b(seat|seats|upgrade|comfort\+?|comfort plus|first class|aisle|window|assignment)\b"),
 ("loyalty_account",   r"\b(skymiles|sky miles|medallion|miles|points|skyclub|sky club|lounge|diamond|platinum|gold member|status)\b"),
 ("tech_support",      r"\b(app|website|wifi|wi.?fi|online check|log ?in|error|the site|the page|won.?t load|doesn.?t work)\b"),
 ("airport_gate",      r"\b(gate|boarding|board|terminal|check.?in counter|tsa|kiosk|gate agent)\b"),
]
PRAISE = re.compile(r"\b(thank|thanks|love|great|best|awesome|amazing|appreciate|welcome|congrat)\b", re.I)

def weak_label(t: str) -> str:
    tl = t.lower()
    for name, pat in RULES:
        if re.search(pat, tl):
            return name
    # no issue keyword -> praise/chit-chat / non-actionable
    if PRAISE.search(tl) or len(tl.split()) <= 4:
        return "other_non_actionable"
    return "other_non_actionable"

def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here,"..","data","delta_customer_messages.csv"))
    ap.add_argument("--out",  default=os.path.join(here,"..","data","golden_to_label.csv"))
    ap.add_argument("--per_class", type=int, default=25)   # floor per actionable class
    ap.add_argument("--other_n",   type=int, default=45)   # non-actionable quota
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df["clean"] = df["clean"].fillna("").astype(str)
    df = df[df["clean"].str.len() >= 5].copy()
    df["suggested_intent"] = df["clean"].map(weak_label)

    dist = df["suggested_intent"].value_counts()
    print("Estimated intent distribution (weak keyword labels, NOT ground truth):")
    for k,v in dist.items():
        print(f"  {k:22s} {v:6,d}  ({100*v/len(df):4.1f}%)")

    # stratified candidate sample
    picks = []
    for name in [r[0] for r in RULES]:
        sub = df[df["suggested_intent"]==name]
        picks.append(sub.sample(min(a.per_class, len(sub)), random_state=a.seed))
    other = df[df["suggested_intent"]=="other_non_actionable"]
    picks.append(other.sample(min(a.other_n, len(other)), random_state=a.seed))
    gold = pd.concat(picks).drop_duplicates("tweet_id").sample(frac=1, random_state=a.seed)

    gold_out = gold[["tweet_id","clean","suggested_intent"]].rename(columns={"clean":"text"})
    gold_out["intent"] = ""          # <- you fill (correct the suggestion)
    gold_out["decision"] = ""        # <- you fill: auto | escalate
    gold_out["notes"] = ""           # <- you fill: why
    gold_out.to_csv(a.out, index=False)
    print(f"\nWrote {len(gold_out)} candidates to label -> data/golden_to_label.csv")
    print("Next: open it, correct 'intent', set 'decision' (auto/escalate), add 'notes'.")

if __name__ == "__main__":
    main()
