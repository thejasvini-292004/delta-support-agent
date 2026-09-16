"""Diagnose the OpenAI key: is it valid, and does a call work?"""
import warnings, traceback; warnings.filterwarnings("ignore")
from openai import OpenAI
from llm import _key, MODEL
key=_key()
print("key prefix:", (key[:6]+"…") if key else None, "| length:", len(key) if key else 0)
print("(a valid OpenAI key starts with 'sk-' )\n")
client=OpenAI(api_key=key)
print("--- trying a real call ---")
for model in [MODEL, "gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"]:
    try:
        r=client.chat.completions.create(model=model, max_tokens=10,
            messages=[{"role":"user","content":"Reply with exactly: OK"}])
        print(f"  {model}: OK -> {r.choices[0].message.content!r}"); break
    except Exception as e:
        print(f"  {model}: FAILED -> {type(e).__name__}: {str(e)[:160]}")
