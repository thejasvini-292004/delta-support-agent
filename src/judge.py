"""LLM-as-judge for reply quality + judge-vs-human agreement (Cohen's kappa)."""
import json, re
from sklearn.metrics import cohen_kappa_score
import llm

_SYS = ("You are a strict QA reviewer for Delta support replies. Given the customer message, "
        "the retrieved past resolutions the draft was grounded on, and the draft reply, score it. "
        'Reply ONLY as JSON: {"grounded":1-5,"correct":1-5,"tone":1-5,'
        '"should_escalate":true/false,"reason":"one line"}. '
        "grounded=every claim supported by the retrieved context; correct=actually helpful/accurate; "
        "tone=empathetic, concise, on-brand; should_escalate=true if a human should handle it.")

def judge_reply(customer, context, reply):
    out = llm.chat(_SYS, f"Customer: {customer}\n\nRetrieved context:\n{context}\n\nDraft reply:\n{reply}", max_tokens=120)
    m=re.search(r"\{.*\}", out, re.S)
    try: return json.loads(m.group(0))
    except Exception: return {"grounded":0,"correct":0,"tone":0,"should_escalate":True,"reason":"parse_error"}

def kappa(human_labels, judge_labels):
    """Cohen's kappa between human and LLM-judge escalate/auto decisions."""
    return cohen_kappa_score(human_labels, judge_labels)
