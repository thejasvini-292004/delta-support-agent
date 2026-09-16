"""The end-to-end Delta support agent: classify -> retrieve -> draft (grounded) -> route.

Design (evidence-based): intent classification ALWAYS uses TF-IDF+LogReg because it beat
the free LLM few-shot on our golden set (macro-F1 0.43 vs the small LLM) and is free,
deterministic and reliable. The LLM is used ONLY to DRAFT replies, and only when a working
key is present; otherwise replies are grounded on the closest historical Delta reply.
"""
import os, re, argparse
from taxonomy import AUTO_OK, RISK_WORDS, CANON_INTENTS, load_golden, negative_sentiment
from retriever import Retriever
import classifiers, llm

GROUNDING_MIN = 0.12          # below this, retrieval is essentially unrelated
GROUNDING_GOOD = 0.38         # below this, the nearest past reply is too dissimilar to send as an answer

class SupportAgent:
    def __init__(self):
        self.retr = Retriever()
        # --- intent classifier: ALWAYS TF-IDF+LogReg (best + free + reliable) ---
        g = load_golden()
        self.clf = classifiers.build_production_classifier(list(g["text"]), list(g["intent_canon"]))
        # --- LLM: used ONLY for drafting replies, and only if a working key is present ---
        self.use_llm = llm.available()
        if self.use_llm:
            ok, info = llm.selftest()
            if not ok:
                print(f"[!] LLM present but the call failed — drafting replies from retrieval instead.\n    reason: {info}")
                self.use_llm = False

    # High-precision overrides: brand terms that are essentially never anything else.
    # A linear bag-of-words model can be pulled off by a co-occurring word (e.g. "SkyMiles ...
    # checked bag" -> baggage). These unambiguous tokens win when present.
    _OVERRIDES = [
        ("loyalty", re.compile(r"\b(skymiles|sky\s?miles|medallion|skyclub|sky\s?club)\b", re.I)),
    ]

    def classify(self, text):
        """Always TF-IDF, with a small high-precision override for unambiguous brand terms."""
        proba = self.clf.predict_proba([text])[0]
        i = proba.argmax()
        pred, conf = self.clf.classes_[i], float(proba[i])
        for label, pat in self._OVERRIDES:
            if pat.search(str(text)) and pred != label:
                classes = list(self.clf.classes_)
                p_label = float(proba[classes.index(label)]) if label in classes else 0.0
                return label, max(p_label, 0.75)
        return pred, conf

    def draft_reply(self, text, intent, hits):
        if intent == "other_non_actionable":
            return ("Thanks so much for reaching out! We appreciate you. "
                    "If there's anything we can help with, just let us know. ✈️", "acknowledgement")
        top = hits[0]["score"] if hits else 0.0
        if top < GROUNDING_GOOD:
            return ("I want to make sure I get this right — I don't have a close enough past case to "
                    "answer this confidently. Let me connect you with a Delta specialist who can help.",
                    "low_grounding")
        if self.use_llm:
            ctx = "\n".join(f"- Past issue: {h['customer_msg']}\n  Delta replied: {h['brand_reply']}"
                            for h in hits)
            sys = ("You draft Delta customer-support replies. Ground your reply ONLY in the past "
                   "Delta replies provided. Be concise, empathetic, on-brand. If they don't cover "
                   "the issue, say you'll connect them to a specialist. Do not invent policies.")
            reply = llm.chat(sys, f"Customer: {text}\n\nSimilar past resolutions:\n{ctx}\n\nDraft the reply:")
            if reply.strip():
                return reply.strip(), "llm_grounded"
            # LLM failed at call time (e.g. credits) -> fall back to retrieval, don't send blank
        # no-LLM (or LLM failed): ground on the closest historical resolution
        return hits[0]["brand_reply"], "retrieved_grounded"

    ESC_TEMPLATES = {
        "flight_disruption": "I'm sorry your travel was disrupted — I know how stressful that is. I'm connecting you with a Delta specialist who can pull up your booking and help get you moving.",
        "baggage": "I'm sorry for the trouble with your baggage. I'm looping in a specialist who can help locate your bag and assist you directly.",
        "refund_compensation": "I understand, and I want to get this looked into for you. I'm connecting you with a specialist who can review the charge and any refund on your account.",
        "booking_fare": "I can help get that sorted — I'm connecting you with a Delta specialist who can securely make that change to your booking.",
        "seating": "Let me bring in a specialist who can access your reservation and help with your seat.",
        "safety_legal": "Thank you for flagging this — it's important, and I'm escalating it to the appropriate Delta team right away.",
        "service_feedback": "Thank you for taking the time to share this — we take your feedback seriously. I'm connecting you with a specialist who can follow up.",
        "loyalty": "Let me connect you with a specialist who can look into your SkyMiles and account details.",
        "technology": "Sorry you're running into that — I'm connecting you with a specialist who can dig into the issue on your account.",
        "airport_gate": "Let me get you to our airport team who can help with this right now.",
    }
    DEFAULT_ESC = "Thanks for reaching out — I'm looping in a Delta specialist to make sure this is handled properly. They'll follow up shortly."

    def _escalation_reply(self, text, intent, negative):
        # tailored to the message; uses the LLM when available, else contextual templates
        if self.use_llm:
            tone = ("The customer is upset: sincerely apologize for the inconvenience and say their "
                    "feedback will be shared with the team."
                    if negative else
                    "The customer is NOT complaining \u2014 do NOT apologize and do NOT invent any "
                    "frustration, problem, or bad experience. Stay neutral and helpful.")
            sysp = ("You are a Delta support agent. In 1-2 warm sentences, acknowledge the customer's "
                    "SPECIFIC question or request in your own words. " + tone + " Then say you're "
                    "connecting them with a Delta specialist who can give the exact details / follow up. "
                    "Do NOT promise refunds or outcomes, do NOT invent policies, numbers, or facts, and "
                    "do NOT assume anything the customer did not say.")
            r = llm.chat(sysp, f"Customer message: {text}")
            if r.strip():
                return r.strip()
        if negative and intent in ("service_feedback", "other_non_actionable"):
            return ("I'm truly sorry to hear you didn't have a good experience — please accept our apologies "
                    "for the inconvenience. We'll take your feedback into consideration and share it with the "
                    "team, and I'm connecting you with a Delta specialist who can follow up with you.")
        msg = self.ESC_TEMPLATES.get(intent, self.DEFAULT_ESC)
        if negative:
            msg = "I'm sorry for the frustration this has caused. " + msg
        return msg

    def route(self, text, intent, confidence, hits, negative=False):
        reasons=[]
        tl=text.lower()
        top = hits[0]["score"] if hits else 0.0
        if negative:                         reasons.append("negative sentiment (dissatisfied customer)")
        if any(w in tl for w in RISK_WORDS): reasons.append("risk/complaint/financial language")
        if intent not in AUTO_OK:            reasons.append(f"intent '{intent}' is account/transaction-specific")
        if confidence < 0.40:                reasons.append(f"low intent confidence ({confidence:.2f})")
        if top < GROUNDING_GOOD:             reasons.append(f"weak grounding in historical replies (top={top:.2f})")
        decision = "escalate" if reasons else "auto"
        reason = "; ".join(reasons) if reasons else f"known intent '{intent}', confident, well-grounded"
        return decision, reason

    def handle(self, text):
        intent, conf = self.classify(text)
        neg = negative_sentiment(text)
        if neg and intent == "other_non_actionable":
            intent = "service_feedback"      # a complaint is feedback, not "non-actionable"
        hits = self.retr.retrieve(text, k=3)
        decision, reason = self.route(text, intent, conf, hits, neg)
        if decision == "escalate":
            reply, mode = self._escalation_reply(text, intent, neg), "escalation"
        else:
            reply, mode = self.draft_reply(text, intent, hits)   # LLM only runs on auto-handled msgs
        return {"text":text, "intent":intent, "confidence":round(conf,3),
                "decision":decision, "reason":reason, "reply_mode":mode,
                "reply":reply, "top_grounding_score":round(hits[0]["score"],3) if hits else 0.0,
                "sources":[h["brand_reply"] for h in hits]}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--msg", default=None); a=ap.parse_args()
    agent=SupportAgent()
    print("Intent classifier: TF-IDF (always)  |  LLM reply-drafting:", agent.use_llm)
    demos=[a.msg] if a.msg else [
        "my flight DL123 got cancelled and I need to get home tonight, help!",
        "how many SkyMiles do I need for a free checked bag?",
        "You are the best airline ever, thank you for the smooth flight! ❤️",
        "you charged my card twice for the same ticket, I want a refund now",
    ]
    import json
    for m in demos:
        print(json.dumps(agent.handle(m), indent=2)[:900]); print("-"*60)
