"""Step 6 — evaluation harness. Runs all three tasks on the golden set with baselines,
writes metrics to reports/. LLM parts (Gemini) run only if a key is set AND reachable.
LLM calls are capped for free-tier rate limits; use --full for the complete run."""
import os, json, argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, accuracy_score)
from taxonomy import load_golden, AUTO_OK, RISK_WORDS, CANON_INTENTS, negative_sentiment
from agent import GROUNDING_MIN, GROUNDING_GOOD
from retriever import Retriever
import classifiers, llm

HERE=os.path.dirname(os.path.abspath(__file__)); REP=os.path.join(HERE,"..","reports")
os.makedirs(REP, exist_ok=True)
def w(name, s): open(os.path.join(REP,name),"w").write(s); print(f"  wrote reports/{name}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run LLM on the full set (slow, may hit rate limits)")
    ap.add_argument("--llm_n", type=int, default=80, help="LLM classify subsample size (ignored with --full)")
    ap.add_argument("--judge_n", type=int, default=15)
    ap.add_argument("--agree_n", type=int, default=40)
    a=ap.parse_args()

    g = load_golden(); y=list(g["intent_canon"]); texts=list(g["text"])
    use_llm = llm.available()
    if use_llm:
        ok, info = llm.selftest()
        if not ok:
            print(f"[!] LLM call failed — running WITHOUT the LLM (baselines only).\n    reason: {info}")
            use_llm = False
    print(f"golden: {len(g)} rows | {len(set(y))} canonical intents | LLM: {use_llm}")

    # ---------- TASK 1: INTENT ----------
    print("\n== TASK 1: intent classification ==")
    rows=[]
    maj=classifiers.majority_label(y); mp=[maj]*len(y)
    rows.append(("trivial: majority class", accuracy_score(y,mp), f1_score(y,mp,average='macro',zero_division=0)))
    oof=classifiers.cv_oof_predict(texts,y,k=5)
    rows.append(("simple: TF-IDF+LogReg (5-fold OOF)", accuracy_score(y,oof), f1_score(y,oof,average='macro',zero_division=0)))
    best=oof
    if use_llm:
        idx = list(range(len(texts))) if a.full else list(pd.Series(range(len(texts))).sample(min(a.llm_n,len(texts)),random_state=42))
        yt=[y[i] for i in idx]; lp=[classifiers.llm_classify(texts[i])[0] for i in idx]
        tag = "full" if a.full else f"{len(idx)}-row subsample"
        rows.append((f"main: LLM few-shot ({tag})", accuracy_score(yt,lp), f1_score(yt,lp,average='macro',zero_division=0)))
        if a.full: best=lp
    tab="model,accuracy,macro_f1\n"+"\n".join(f"{n},{acc:.3f},{f:.3f}" for n,acc,f in rows)
    print("  "+tab.replace("\n","\n  ")); w("intent_scores.csv", tab)
    w("intent_classification_report.txt", classification_report(y,oof,zero_division=0))
    w("intent_confusion_matrix.csv", pd.DataFrame(confusion_matrix(y,oof,labels=CANON_INTENTS),
        index=CANON_INTENTS,columns=CANON_INTENTS).to_csv())

    # ---------- TASK 3: ESCALATION (oracle intent -> pure policy quality) ----------
    print("\n== TASK 3: escalation decision ==")
    retr=Retriever()
    def route(text,intent):
        tl=text.lower(); top=retr.retrieve(text,1)[0]["score"]
        esc=(negative_sentiment(text) or any(x in tl for x in RISK_WORDS)
             or (intent not in AUTO_OK) or (top<GROUNDING_GOOD))
        return "escalate" if esc else "auto"
    human=list(g["decision"]); preds=[route(t,i) for t,i in zip(texts,y)]
    def m(name,p):
        acc=accuracy_score(human,p); pr,rc,f1,_=precision_recall_fscore_support(human,p,labels=["auto","escalate"],zero_division=0)
        return f"{name},{acc:.3f},{pr[0]:.3f},{rc[0]:.3f},{pr[1]:.3f},{rc[1]:.3f}"
    lines=["model,accuracy,auto_precision,auto_recall,escalate_precision,escalate_recall",
           m("trivial: escalate-all",["escalate"]*len(human)),
           m("trivial: auto-all",["auto"]*len(human)), m("rules (ours)",preds)]
    print("  "+"\n  ".join(lines)); w("escalation_scores.csv","\n".join(lines))

    # ---------- TASK 2: REPLY QUALITY (LLM judge) + judge-human agreement ----------
    print("\n== TASK 2: reply quality ==")
    from agent import SupportAgent
    agent=SupportAgent()
    sample=g.sample(min(a.judge_n if use_llm else 8,len(g)), random_state=42)
    drafts=[agent.handle(t) for t in sample["text"]]
    w("sample_agent_outputs.json", json.dumps(drafts,indent=2))
    if use_llm:
        import judge
        scored=[judge.judge_reply(d["text"],"\n".join(d["sources"]),d["reply"]) for d in drafts]
        avg={k:round(float(np.mean([s.get(k,0) for s in scored])),2) for k in ["grounded","correct","tone"]}
        sl=g.sample(min(a.agree_n,len(g)),random_state=7)
        jd=["escalate" if judge.judge_reply(t,"",agent.handle(t)["reply"]).get("should_escalate",True) else "auto" for t in sl["text"]]
        kap=judge.kappa(list(sl["decision"]),jd)
        out={"avg_scores":avg,"judge_human_escalation_kappa":round(kap,3),"n_judged":len(scored),"n_agreement":len(jd)}
        print("  ",out); w("reply_judge_summary.json", json.dumps(out,indent=2))
    else:
        print("  [LLM unreachable/no key — wrote retrieval-grounded sample replies only]")
        w("reply_judge_summary.json", json.dumps({"note":"LLM skipped (no key or blocked network)"},indent=2))
    print("\nDONE. See reports/ and REPORT.md")

if __name__=="__main__": main()
