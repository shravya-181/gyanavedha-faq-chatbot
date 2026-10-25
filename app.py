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
div[data-testid="stButton"] > button:hover
