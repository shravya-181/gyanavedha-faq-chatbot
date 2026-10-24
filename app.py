"""
app.py  -  the Streamlit chat page.   Run with:  streamlit run app.py
"""

import streamlit as st

from chatbot.bot import Chatbot
from chatbot.logger import DEFAULT_PATH, read_unanswered

FAQ_PATH = "data/faqs.json"

st.set_page_config(page_title="Institute Assistant", page_icon="🎓")


@st.cache_resource
def load_bot() -> Chatbot:
    """Train the bot once and reuse it (Streamlit reruns this file on every click)."""
    return Chatbot(FAQ_PATH)


bot = load_bot()
name = bot.classifier.business_name

st.title(f"🎓 {name} Assistant")
st.caption("Ask about admissions, fees, courses, exams, hostel, scholarships and more.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Options")
    show_details = st.checkbox("Show how the bot decided", value=False)

    st.header("Unanswered questions")
    rows = read_unanswered(DEFAULT_PATH)
    st.write(f"{len(rows)} logged so far")
    if rows:
        with open(DEFAULT_PATH, "rb") as f:
            st.download_button("Download as CSV", f.read(),
                               file_name="unanswered.csv", mime="text/csv")

# ---------------- Chat history ----------------
# st.session_state keeps data between reruns of this script.
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Hello! Welcome to {name}. How can I help you today?",
        "details": [],
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_details and msg["details"]:
            with st.expander("How I decided"):
                for d in msg["details"]:
                    label = "no answer" if d["is_fallback"] else d["intent"]
                    st.write(f"**{d['question']}** -> {label} "
                             f"(confidence {d['confidence']})")

# ---------------- New message ----------------
user_text = st.chat_input("Type your question here...")
if user_text:
    st.session_state.messages.append(
        {"role": "user", "content": user_text, "details": []})

    result = bot.reply(user_text)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["text"], "details": result["details"]})

    st.rerun()   # redraw the page so the new messages appear