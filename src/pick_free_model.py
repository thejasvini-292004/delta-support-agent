"""Auto-find a FREE OpenRouter model that works with your key, and save it to .env.
Run once on your Mac:  python3 src/pick_free_model.py"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
from openai import OpenAI

c = OpenAI(api_key=llm._key(), base_url="https://openrouter.ai/api/v1")
free = sorted(m.id for m in c.models.list().data if m.id.endswith(":free"))
print(f"{len(free)} free models listed for your key. Testing for a working one...\n")

# prefer capable instruct/chat models first
pref = ["llama-3.3","llama-3.1-70b","qwen-2.5-72b","deepseek-chat","deepseek-r1",
        "gemini-2.0-flash","mistral","gemma-2","qwen"]
def rank(x): return min([i for i,p in enumerate(pref) if p in x.lower()] or [99])
free.sort(key=rank)

winner=None
for m in free[:15]:
    try:
        r=c.chat.completions.create(model=m, max_tokens=5,
            messages=[{"role":"user","content":"Reply with: OK"}])
        if (r.choices[0].message.content or "").strip():
            winner=m; print(f"  ✓ WORKS -> {m}"); break
        print(f"  ✗ empty     {m}")
    except Exception as e:
        print(f"  ✗ {m}: {str(e)[:60]}")

if winner:
    envp=os.path.join(os.path.dirname(__file__),"..",".env")
    txt=open(envp).read() if os.path.exists(envp) else ""
    if re.search(r"(?m)^HIVER_MODEL=", txt):
        txt=re.sub(r"(?m)^HIVER_MODEL=.*$", f"HIVER_MODEL={winner}", txt)
    else:
        txt += f"\nHIVER_MODEL={winner}\n"
    open(envp,"w").write(txt)
    print(f"\nSaved HIVER_MODEL={winner} to .env.")
    print("Now run:\n  python3 src/llm.py            # should print reply: 'OK'")
    print("  python3 src/evaluate.py --llm_n 20 --judge_n 6 --agree_n 10")
else:
    print("\nNo free model worked (free tier likely exhausted/rate-limited).")
    print("Add ~$5 at openrouter.ai -> Credits, then use the default paid model:")
    print("  python3 src/evaluate.py --llm_n 20 --judge_n 6 --agree_n 10")
