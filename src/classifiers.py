"""Intent classifiers: majority baseline, TF-IDF+LogReg (simple baseline), LLM few-shot (main)."""
import json, re, numpy as np
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from taxonomy import CANON_INTENTS
import llm

def majority_label(labels): return Counter(labels).most_common(1)[0][0]

def build_tfidf_logreg():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0)),
    ])

def cv_oof_predict(texts, labels, k=5, seed=42):
    """Out-of-fold predictions for a fair estimate on a small labelled set."""
    texts=np.array(texts); labels=np.array(labels)
    oof=np.empty(len(texts), dtype=object)
    skf=StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr,te in skf.split(texts, labels):
        m=build_tfidf_logreg().fit(texts[tr], labels[tr])
        oof[te]=m.predict(texts[te])
    return list(oof)

# ---- LLM few-shot classifier ----
_SYS = ("You are an intent classifier for Delta Air Lines customer-support tweets. "
        "Classify the message into exactly one of these intents:\n" +
        ", ".join(CANON_INTENTS) +
        "\nDefinitions: flight_disruption=delay/cancel/missed connection; baggage=bags/luggage/fees; "
        "booking_fare=change/cancel/rebook/standby/fares; refund_compensation=refunds, money back, vouchers; "
        "seating=seat selection/upgrades/cabin; airport_gate=gate/boarding/terminal/check-in; "
        "loyalty=SkyMiles/Medallion/SkyClub/points; technology=app/website/wifi/login errors; "
        "service_feedback=praise or complaint about service/crew with no transaction; "
        "safety_legal=safety, legal, discrimination, injury; "
        "other_non_actionable=praise/jokes/tagging/venting with no request.\n"
        'Reply ONLY as JSON: {"intent": "...", "confidence": 0.0-1.0}.')

def llm_classify(text):
    out = llm.chat(_SYS, text, max_tokens=60)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0)); intent=d.get("intent","other_non_actionable")
        if intent not in CANON_INTENTS: intent="other_non_actionable"
        return intent, float(d.get("confidence",0.5))
    except Exception:
        return "other_non_actionable", 0.3


# ---------------------------------------------------------------------------
# Production classifier: bootstrap volume with weak (keyword) supervision over
# the 44k messages, anchored by the 220 gold labels (upweighted). More data ->
# sharper, more confident, more stable predictions than 220 rows alone.
# ---------------------------------------------------------------------------
import os, re as _re, pandas as _pd
_WEAK = [
 ("safety_legal", r"\b(safety|unsafe|legal|lawsuit|sue|attorney|lawyer|discriminat|racist|assault|injur|emergency|medical|wheelchair|harass)\b"),
 ("refund_compensation", r"\b(refund|compensat|voucher|money back|reimburse|credit back|charge.?back)\b"),
 ("loyalty", r"\b(skymiles|sky miles|medallion|miles|points|skyclub|sky club|lounge|diamond|platinum|gold member|elite|status)\b"),
 ("flight_disruption", r"\b(delay|delayed|cancel|cancell?ed|cancellation|missed? connection|diverted|stranded|stuck|reroute)\b"),
 ("seating", r"\b(seat|seats|upgrade|comfort\+?|first class|aisle|window|cabin)\b"),
 ("baggage", r"\b(bag|bags|baggage|luggage|suitcase|carry.?on|checked bag|lost bag|gate check)\b"),
 ("booking_fare", r"\b(change my flight|reschedul|standby|basic economy|fare|itinerary|change fee|rebook|book(ing)?)\b"),
 ("airport_gate", r"\b(gate|boarding|board|terminal|check.?in counter|tsa|kiosk)\b"),
 ("technology", r"\b(app|website|wifi|wi.?fi|online check|log ?in|error|the site|the page|crash|glitch)\b"),
 ("service_feedback", r"\b(thank|thanks|love|great|best|awesome|amazing|appreciate|wonderful|kudos|terrible|awful|rude|worst|disappointed|horrible)\b"),
]
def canonical_weak_label(t):
    tl=str(t).lower()
    for name,pat in _WEAK:
        if _re.search(pat, tl): return name
    return "other_non_actionable"

def _weak_corpus(n=7000, seed=42):
    here=os.path.dirname(os.path.abspath(__file__))
    df=_pd.read_csv(os.path.join(here,"..","data","delta_customer_messages.csv"))
    df["clean"]=df["clean"].fillna("").astype(str)
    df=df[df["clean"].str.len()>=5]
    df=df.sample(min(n,len(df)), random_state=seed)
    return list(df["clean"]), [canonical_weak_label(t) for t in df["clean"]]

def build_production_classifier(golden_texts, golden_labels, weak_n=7000, gold_weight=10):
    Xw,yw=_weak_corpus(weak_n)
    X=list(Xw)+list(golden_texts)*gold_weight      # upweight the TRUE gold labels
    y=list(yw)+list(golden_labels)*gold_weight
    pipe=Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True, max_features=30000)),
        ("clf",LogisticRegression(max_iter=2000, C=8.0)),   # sharper (no balanced flattening; volume covers it)
    ])
    return pipe.fit(X,y)
