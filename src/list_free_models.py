"""List OpenRouter models your key can use for free right now."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
from openai import OpenAI
c = OpenAI(api_key=llm._key(), base_url="https://openrouter.ai/api/v1")
ids = [m.id for m in c.models.list().data]
free = sorted(i for i in ids if i.endswith(":free"))
print(f"{len(free)} free models available to your key:\n")
for i in free: print("  ", i)
print('\nPick one, then run e.g.:')
print('  HIVER_MODEL="<slug>" python3 src/evaluate.py --llm_n 20 --judge_n 6 --agree_n 10')
