"""Simple chat UI for the Delta support agent.  Run:  streamlit run app.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import streamlit as st
from agent import SupportAgent
import llm

st.set_page_config(page_title="Delta Support Agent", page_icon="✈️", layout="centered")

@st.cache_resource(show_spinner="Loading the agent (indexing historical replies)…")
def get_agent():
    return SupportAgent()
agent = get_agent()

st.title("✈️ Delta Support Agent")
replies = f"LLM ({llm.MODEL})" if agent.use_llm else "retrieval"
st.caption(f"Brand: Delta  ·  Intent: TF-IDF  ·  Replies: {replies}")

EXAMPLES = [
    "My flight DL123 got cancelled and I need to get home tonight!",
    "How many SkyMiles do I need for a free checked bag?",
    "You charged my card twice for the same ticket, I want a refund.",
    "You're the best airline ever, thank you! ❤️",
    "My bag never showed up at baggage claim in ATL.",
]

with st.sidebar:
    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
    st.caption("Every reply shows the predicted intent, the auto-handle vs escalate "
               "decision with a reason, and the past Delta replies it was grounded on.")

if "history" not in st.session_state: st.session_state.history = []

def render(res):
    with st.chat_message("assistant", avatar="✈️"):
        st.write(res["reply"])
        badge = "🟢 Auto-handled" if res["decision"] == "auto" else "🔴 Escalated to a human"
        st.markdown(f"**{badge}**  ·  intent `{res['intent']}` ({res['confidence']:.0%} conf)")
        with st.expander("🔎 How the agent decided"):
            st.markdown(f"**Decision:** `{res['decision']}` — {res['reason']}")
            st.markdown(f"**Intent:** `{res['intent']}`  ·  confidence {res['confidence']:.2f}")
            st.markdown(f"**Top grounding score:** {res['top_grounding_score']:.2f}  "
                        f"(higher = closer past case)")
            st.markdown("**Retrieved past Delta replies (grounding):**")
            for s in res["sources"]:
                st.markdown(f"> {s}")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["text"])
    render(turn["res"])

prompt = st.chat_input("Type a customer message…")
if st.session_state.get("pending"):
    prompt = st.session_state.pending
    st.session_state.pending = None

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.spinner("Thinking…"):
        res = agent.handle(prompt)
    render(res)
    st.session_state.history.append({"text": prompt, "res": res})
