"""LLM wrapper — auto-detects the provider from the key and speaks to all of them
through one interface: available() / chat() / selftest().

  hf_...   -> Hugging Face router (OpenAI-compatible)
  sk-or... -> OpenRouter        (OpenAI-compatible)
  sk-...   -> OpenAI
  else     -> Google Gemini     (AIza... / AQ... keys)
"""
import os, re, time, warnings
warnings.filterwarnings("ignore")
_here = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv():
    for fn in ("../.env", "../API_KEY_HERE.txt"):
        path = os.path.join(_here, *fn.split("/"))
        if not os.path.exists(path): continue
        for line in open(path):
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v = line.split("=",1); v=v.strip().strip('"\'')
                if v: os.environ.setdefault(k.strip(), v)
_load_dotenv()

def _key():
    for var in ("LLM_API_KEY","HF_TOKEN","OPENAI_API_KEY","OPENROUTER_API_KEY",
                "GEMINI_API_KEY","GOOGLE_API_KEY","OPEN_API_KEY"):
        v=os.environ.get(var)
        if v: return v
    return None

def _provider():
    if os.environ.get("LLM_PROVIDER"): return os.environ["LLM_PROVIDER"].lower()
    k=_key() or ""
    if k.startswith("hf_"):   return "hf"
    if k.startswith("sk-or"): return "openrouter"
    if k.startswith("sk-"):   return "openai"
    return "gemini"

_DEFAULTS = {"openai":"gpt-4o-mini",
             "openrouter":"meta-llama/llama-3.3-70b-instruct",
             "hf":"meta-llama/Llama-3.1-8B-Instruct",
             "gemini":"gemini-3.6-flash"}
_BASE = {"openrouter":"https://openrouter.ai/api/v1",
         "hf":"https://router.huggingface.co/v1"}     # openai -> None (default)

def _model(): return os.environ.get("HIVER_MODEL") or _DEFAULTS[_provider()]
MODEL=_model()
_last_error=None

def available()->bool:
    if not _key(): return False
    try:
        if _provider()=="gemini": import google.generativeai  # noqa
        else: import openai  # noqa  (openai/openrouter/hf all use the openai SDK)
        return True
    except Exception:
        return False

def _gemini_text(r):
    try:
        if r.text: return r.text.strip()
    except Exception:
        pass
    try:
        return "".join(getattr(pt,"text","") for pt in r.candidates[0].content.parts).strip()
    except Exception:
        return ""

def chat(system:str, user:str, max_tokens=400, temperature=0.0)->str:
    global _last_error
    p=_provider()
    for attempt in range(3):
        try:
            if p=="gemini":
                import google.generativeai as genai
                genai.configure(api_key=_key())
                safe=[{"category":c,"threshold":"BLOCK_NONE"} for c in
                    ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH",
                     "HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
                m=genai.GenerativeModel(_model(), system_instruction=system)
                r=m.generate_content(user, generation_config={"temperature":temperature,
                    "max_output_tokens":max(max_tokens,1024)}, safety_settings=safe,
                    request_options={"timeout":30})
                txt=_gemini_text(r)
                if txt: return txt
                try: fr=r.candidates[0].finish_reason
                except Exception: fr=None
                _last_error=f"gemini returned no text (finish_reason={fr})"; return ""
            else:
                from openai import OpenAI
                client=OpenAI(api_key=_key(), base_url=_BASE.get(p), timeout=30, max_retries=0)
                r=client.chat.completions.create(model=_model(), temperature=temperature,
                    max_tokens=max_tokens, messages=[{"role":"system","content":system},
                                                     {"role":"user","content":user}])
                return (r.choices[0].message.content or "").strip()
        except Exception as e:
            es=str(e); _last_error=f"{type(e).__name__}: {es[:300]}"
            if p=="gemini":
                mm=re.search(r"use\s+models/([A-Za-z0-9.\-]+)", es)   # "...use models/X..."
                if mm and mm.group(1)!=_model() and attempt<2:
                    os.environ["HIVER_MODEL"]=mm.group(1); continue
            if any(x in es.lower() for x in ["429","rate","quota","exhaust","overloaded","503"]) and attempt<2:
                time.sleep(3*(attempt+1)); continue
            return ""
    return ""

def selftest():
    r=chat("You are a test.","Reply with exactly: OK", max_tokens=10)
    return (bool(r.strip()), r if r.strip() else (_last_error or "empty response"))

if __name__=="__main__":
    print("provider:", _provider(), "| key present:", bool(_key()), "| model:", _model())
    ok,info=selftest()
    if ok: print("reply:", repr(info))
    else:  print("\nLLM CALL FAILED. Real reason:\n  ", info)
