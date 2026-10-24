"""
app.py  -  the Streamlit chat page.   Run with:  streamlit run app.py
"""

import json
import re
import time

import streamlit as st

from chatbot.bot import Chatbot
from chatbot.logger import DEFAULT_PATH, log_feedback, read_feedback, read_unanswered

FAQ_PATH = "data/faqs.json"
SKIP_INTENTS = {"greeting", "goodbye", "thanks"}

st.set_page_config(page_title="Institute Assistant", page_icon="🎓")

# ---------------- Look and feel ----------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 60%, #3b82f6 100%);
    color: white; padding: 22px 26px; border-radius: 16px; margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(29, 78, 216, 0.25);
}
.hero-title { font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
.hero-sub { font-size: 0.95rem; opacity: 0.9; margin-top: 4px; }
.hero-pill {
    display: inline-block; background: rgba(255,255,255,0.18);
    padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; margin-top: 10px;
}
.chip-label { font-size: 0.85rem; color: #475569; margin: 4px 0; }
div[data-testid="stButton"] > button {
    border-radius: 999px; border: 1px solid #93b4f5; background: white;
    color: #1d4ed8; font-size: 0.85rem; padding: 2px 10px;
}
div[data-testid="stButton"] > button:hover { background: #e8eefc; border-color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


# ---------------- Load the bot and the FAQ settings ----------------
@st.cache_resource
def load_bot() -> Chatbot:
    """Train the bot once and reuse it (Streamlit reruns this file on every click)."""
    return Chatbot(FAQ_PATH)


@st.cache_data
def load_settings() -> dict:
    """Read the optional display settings from faqs.json."""
    with open(FAQ_PATH, encoding="utf-8") as f:
        data = json.load(f)
    names = [i["name"] for i in data["intents"] if i["name"] not in SKIP_INTENTS]
    default_chips = [n.replace("_", " ").title() for n in names[:6]]
    return {
        "tagline": data.get("tagline", "Admissions, fees, hostel, placements and more"),
        "suggestions": data.get("suggestions", default_chips),
        "contact": data.get("contact", ""),
    }


bot = load_bot()
settings = load_settings()
name = bot.classifier.business_name

# ---------------- Header ----------------
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🎓 {name}</div>
  <div class="hero-sub">{settings['tagline']}</div>
  <div class="hero-pill">● Online - replies instantly</div>
</div>
""", unsafe_allow_html=True)


# ---------------- Helpers ----------------
def stream_words(text: str):
    """Yield the reply a few characters at a time, so it looks like typing."""
    for piece in re.split(r"(\s+)", text):
        yield piece
        time.sleep(0.015)


def ask(text: str) -> None:
    """Send one customer message to the bot and store both sides of the chat."""
    result = bot.reply(text)
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.messages.append({
        "role": "assistant", "content": result["text"], "details": result["details"],
        "question": text, "animate": True,
    })


def save_feedback(index: int) -> None:
    """Called when someone clicks thumbs up/down under an answer."""
    msg = st.session_state.messages[index]
    rating = st.session_state[f"fb_{index}"]      # 1 = thumbs up, 0 = thumbs down
    if rating is not None:
        log_feedback(msg["question"], msg["content"], "up" if rating == 1 else "down")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    if settings["contact"]:
        st.write(settings["contact"])
    show_details = st.checkbox("Show how the bot decided", value=False)
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

    st.header("Activity")
    feedback = read_feedback()
    st.write(f"{len(read_unanswered(DEFAULT_PATH))} unanswered questions logged")
    st.write(f"{sum(f['rating'] == 'up' for f in feedback)} 👍  "
             f"{sum(f['rating'] == 'down' for f in feedback)} 👎")

# ---------------- Chat state ----------------
# st.session_state keeps data between reruns of this script.
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Hello! 👋 Welcome to **{name}**. I can help with admissions, fees, "
                   "courses, hostel and more. Pick a topic below or type your question.",
        "details": [], "question": "", "animate": False,
    }]

# ---------------- Quick-question chips ----------------
st.markdown('<div class="chip-label">Quick questions</div>', unsafe_allow_html=True)
chips = settings["suggestions"]
columns = st.columns(3)
for n, chip in enumerate(chips):
    if columns[n % 3].button(chip, key=f"chip_{n}", width="stretch"):
        ask(chip)

# ---------------- New typed message ----------------
typed = st.chat_input("Type your question here...")
if typed:
    ask(typed)

# ---------------- Show the conversation ----------------
for i, msg in enumerate(st.session_state.messages):
    avatar = "🎓" if msg["role"] == "assistant" else "🙂"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("animate"):
            st.write_stream(stream_words(msg["content"]))
            msg["animate"] = False
        else:
            st.write(msg["content"])

        if msg["role"] == "assistant" and i > 0:
            st.feedback("thumbs", key=f"fb_{i}", on_change=save_feedback, args=(i,))
            if show_details and msg["details"]:
                with st.expander("How I decided"):
                    for d in msg["details"]:
                        label = "no answer" if d["is_fallback"] else d["intent"]
                        st.write(f"**{d['question']}** -> {label} "
                                 f"(confidence {d['confidence']})")
