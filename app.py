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

st.set_page_config(page_title="Institute Help Desk", page_icon="🎓",
                   initial_sidebar_state="collapsed")

# ---------------- Look and feel ----------------
# Idea: the woven gold border of a Mysore silk saree (zari) + the letters ज्ञ.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Hanken+Grotesk:wght@400;500;600&family=Noto+Serif+Devanagari:wght@600&display=swap');

:root {
  --green: #0F4C45;  --green-2: #17695E;  --gold: #C9A227;
  --mist: #F3F6F4;   --ink: #13231F;      --muted: #4C625B;  --line: #D5E0DA;
}
html, body, .stApp, .stMarkdown, p, button, input, textarea {
  font-family: 'Hanken Grotesk', 'Segoe UI', system-ui, sans-serif;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 1.1rem; padding-bottom: 6rem; }

/* zari border */
.zari {
  height: 14px; border-radius: 3px;
  background:
    linear-gradient(135deg, var(--gold) 25%, transparent 25%) -7px 0 / 14px 14px,
    linear-gradient(225deg, var(--gold) 25%, transparent 25%) -7px 0 / 14px 14px,
    linear-gradient(315deg, var(--gold) 25%, transparent 25%) 0 0 / 14px 14px,
    linear-gradient(45deg,  var(--gold) 25%, var(--green) 25%) 0 0 / 14px 14px;
}

/* masthead */
.masthead { display: flex; align-items: center; gap: 18px; padding: 22px 2px 4px; }
.mono {
  flex: none; width: 60px; height: 60px; border-radius: 50%;
  display: grid; place-items: center; background: var(--green); color: var(--gold);
  font-family: 'Noto Serif Devanagari', 'Nirmala UI', 'Mangal', serif;
  font-size: 1.55rem; line-height: 1;
  box-shadow: 0 0 0 3px var(--mist), 0 0 0 5px var(--gold);
}
.m-title {
  font-family: 'Fraunces', Georgia, serif; font-weight: 700; color: var(--green);
  font-size: clamp(1.45rem, 5.2vw, 2rem); line-height: 1.1; letter-spacing: -0.01em;
}
.m-sub { color: var(--muted); font-size: 0.95rem; margin-top: 5px; }
.status { color: var(--muted); font-size: 0.84rem; margin: 10px 2px 18px; }
.dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: #2FA56B; margin-right: 8px; animation: pulse 2.2s ease-out infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(47,165,107,.55); }
  70%, 100% { box-shadow: 0 0 0 9px rgba(47,165,107,0); }
}
@media (prefers-reduced-motion: reduce) { .dot { animation: none; } }

/* chat messages */
[data-testid="stChatMessage"] {
  background: #fff; border: 1px solid var(--line); border-left: 3px solid var(--gold);
  border-radius: 4px 18px 18px 18px; padding: 14px 18px; margin-bottom: 12px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse; background: var(--green); border: 1px solid var(--green);
  border-radius: 18px 4px 18px 18px; width: fit-content; max-width: 84%; margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: #fff; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"] p { line-height: 1.6; font-size: 0.98rem; }

/* question list */
.q-label { font-weight: 600; color: var(--ink); font-size: 0.95rem; margin: 2px 0 8px; }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 10px; }
[data-testid="stColumn"] { flex: 1 1 calc(50% - 10px) !important; min-width: calc(50% - 10px) !important; }
[data-testid="stButton"] { width: 100%; }
div[data-testid="stButton"] > button {
  width: 100%; justify-content: flex-start; text-align: left;
  background: #fff; color: var(--ink); font-weight: 500; font-size: 0.93rem;
  border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px;
  transition: border-color .15s, background .15s, transform .15s;
}
div[data-testid="stButton"] > button { white-space: normal; height: auto; min-height: 44px; }
div[data-testid="stButton"] > button > div { width: 100%; display: flex; justify-content: flex-start; }
div[data-testid="stButton"] > button p {
  text-align: left; white-space: normal; overflow: visible; text-overflow: clip;
}
div[data-testid="stButton"] > button:hover {
  border-color: var(--gold); background: #FFFCF0; color: var(--green); transform: translateY(-1px);
}
div[data-testid="stButton"] > button:focus-visible { outline: 2px solid var(--green); outline-offset: 2px; }

/* input */
[data-testid="stChatInput"] { border-radius: 16px; border: none; background: #fff;
  box-shadow: 0 0 0 1.5px var(--green), 0 6px 18px rgba(15, 76, 69, .10); }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: #fff; }
</style>
""", unsafe_allow_html=True)

ICONS = {
    "admission": "📝", "fee": "💳", "hostel": "🏠", "course": "📚", "exam": "🗓️",
    "scholarship": "🎓", "placement": "💼", "contact": "📞", "transport": "🚌",
    "bus": "🚌", "certificate": "📄", "library": "📖",
}


def icon_for(label: str) -> str:
    for word, icon in ICONS.items():
        if word in label.lower():
            return icon
    return "💬"


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
        "tagline": data.get("tagline", "Admissions, fees, hostel and placements"),
        "suggestions": data.get("suggestions", default_chips),
        "contact": data.get("contact", ""),
        "status": data.get("status", "Help desk is online. Replies in seconds."),
    }


bot = load_bot()
settings = load_settings()
name = bot.classifier.business_name

# ---------------- Header ----------------
st.markdown(f"""
<div class="zari"></div>
<div class="masthead">
  <div class="mono">ज्ञ</div>
  <div>
    <div class="m-title">{name}</div>
    <div class="m-sub">{settings['tagline']}</div>
  </div>
</div>
<div class="status"><span class="dot"></span>{settings['status']}</div>
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
        "content": f"Namaste 🙏 Welcome to **{name}**. Ask me about admissions, fees, "
                   "hostel or placements, or tap a question below.",
        "details": [], "question": "", "animate": False,
    }]

# Handle typed input first, so everything below already knows about it.
typed = st.chat_input("Ask about fees, hostel, admission...")
if typed:
    ask(typed)


# ---------------- Question list ----------------
def question_buttons() -> None:
    columns = st.columns(2)
    for n, chip in enumerate(settings["suggestions"]):
        columns[n % 2].button(f"{icon_for(chip)}  {chip}", key=f"chip_{n}",
                              on_click=ask, args=(chip,), width="stretch")


no_questions_yet = len(st.session_state.messages) <= 1
if no_questions_yet:
    st.markdown('<div class="q-label">Popular questions</div>', unsafe_allow_html=True)
    question_buttons()

# ---------------- Show the conversation ----------------
for i, msg in enumerate(st.session_state.messages):
    # The default "user" avatar lets our CSS tell user and bot messages apart.
    avatar = "📖" if msg["role"] == "assistant" else None
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

if not no_questions_yet:
    with st.expander("More questions"):
        question_buttons()
