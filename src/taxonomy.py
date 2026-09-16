"""Single source of truth: canonical intents, label consolidation, escalation policy."""
import os, pandas as pd

# Consolidate the hand-labels into canonical intents (edit freely & re-run).
# Overlapping labels are merged; everything else is kept as labelled.
LABEL_MAP = {
    "booking_change":"booking_fare", "booking_issue":"booking_fare", "fare_fares":"booking_fare",
    "customer_service":"service_feedback", "onboard_service":"service_feedback",
    # identity maps (kept as-is):
    "flight_disruption":"flight_disruption", "baggage":"baggage", "seating":"seating",
    "refund_compensation":"refund_compensation", "airport_gate":"airport_gate",
    "technology":"technology", "loyalty":"loyalty", "safety_legal":"safety_legal",
    "other_non_actionable":"other_non_actionable",
}
def canon(label:str)->str:
    return LABEL_MAP.get(str(label).strip(), str(label).strip())

# Intents the agent may auto-handle (informational / low-risk). Everything else escalates.
AUTO_OK = {"loyalty","technology","airport_gate","other_non_actionable","service_feedback"}

# Words that force escalation regardless of intent (account/legal/safety/money/anger).
RISK_WORDS = ["refund","compensat","charged","unauthor","lawsuit","legal","sue ","attorney",
    "discriminat","racist","safety","emergency","medical","wheelchair","complaint","manager",
    "cancel my","stranded","missed connection","reschedul","rebook","never flying","worst",
    "disgusted","unacceptable","ridiculous","terrible","awful","horrible","disappointed",
    "angry","furious","rude","hate","pathetic","never again","poor service","frustrated","upset"]

def load_golden(path=None):
    """Read the hand-labelled golden set (xlsx or csv) and add a canonical intent column."""
    here = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        d = os.path.join(here,"..","data")
        for name in ["golden_labeled_delta.xlsx","golden.csv","golden_labeled_delta.csv"]:
            if os.path.exists(os.path.join(d,name)): path=os.path.join(d,name); break
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df["intent"] = df["intent"].astype(str).str.strip()
    df["intent_canon"] = df["intent"].map(canon)
    df["decision"] = df["decision"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    return df

CANON_INTENTS = sorted(set(LABEL_MAP.values()))


# --- negation-aware negative-sentiment detector (dependency-free) -----------
import re as _re
_NEGATORS={"not","no","never","cant","couldnt","wont","didnt","dont","isnt","wasnt","arent","hardly","barely","nothing"}
_POS={"good","great","happy","satisfied","impressed","pleased","nice","enjoy","enjoyed","comfortable",
      "helpful","wonderful","excellent","amazing","love","loved","best","smooth","recommend","pleasant","fun"}
_NEG={"bad","terrible","awful","horrible","disappoint","unhappy","unsatisfied","dissatisfied","poor","worst",
      "rude","unacceptable","ridiculous","frustrat","upset","angry","annoyed","disgust","hate","unimpressed",
      "mediocre","subpar","pathetic","nightmare","worse","complaint","furious","lousy","sucks","suck","fed up"}
def negative_sentiment(text):
    """True if the message expresses dissatisfaction (incl. 'not good', 'not impressed')."""
    tl=str(text).lower()
    if any(n in tl for n in _NEG): return True
    toks=_re.findall(r"[a-z']+", tl.replace("n't"," not"))
    for i,w in enumerate(toks):                      # negated positive -> negative
        if w in _POS and any(t in _NEGATORS for t in toks[max(0,i-3):i]): return True
    return False
