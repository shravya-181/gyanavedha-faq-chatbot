"""
pages/Bot_Doctor.py  -  the Bot Doctor page.
Streamlit shows any file inside a `pages` folder as an extra page in the sidebar.
"""

import pandas as pd
import streamlit as st

from chatbot.doctor import diagnose

st.set_page_config(page_title="Bot Doctor", page_icon="🩺", layout="wide")
st.title("🩺 Bot Doctor")
st.caption("Checks how well the bot understands new questions, and what to fix next.")

if st.button("Re-run diagnosis") or "report" not in st.session_state:
    with st.spinner("Examining the bot..."):
        st.session_state.report = diagnose()

r = st.session_state.report

# ---------------- Summary numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Intents", r["n_intents"])
c2.metric("Training examples", r["n_examples"])
c3.metric("Accuracy on new sentences",
          f"{int(r['accuracy'] * 100)}%" if r["accuracy"] is not None else "n/a")
c4.metric("Current threshold", r["current_threshold"])

# ---------------- What to do next ----------------
st.subheader("What to fix next")
for tip in r["recommendations"]:
    st.write(f"- {tip}")

# ---------------- Per-intent accuracy ----------------
st.subheader("How well does each intent work?")
if r["per_intent"]:
    df = pd.DataFrame(r["per_intent"]).set_index("intent")
    st.bar_chart(df["accuracy"])
    st.caption("Accuracy = how often the bot picks this intent correctly for a sentence "
               "it was NOT trained on (cross-validation).")
    if r["weak"]:
        st.write("**Weak intents**")
        for w in r["weak"]:
            st.write(f"- `{w['intent']}`: " + "; ".join(w["reasons"]))
else:
    st.info("Add at least 2 examples to every intent to enable accuracy testing.")

# ---------------- Confusions and overlaps ----------------
left, right = st.columns(2)
with left:
    st.subheader("Mix-ups")
    if r["confusions"]:
        st.dataframe(pd.DataFrame(r["confusions"]), hide_index=True, width="stretch")
    else:
        st.write("No mix-ups found.")
with right:
    st.subheader("Intents that sound alike")
    if r["overlaps"]:
        st.dataframe(pd.DataFrame(r["overlaps"]), hide_index=True, width="stretch")
    else:
        st.write("No intents share unusually similar wording.")

# ---------------- Threshold ----------------
st.subheader("Is the confidence threshold right?")
if r["thresholds"]:
    t = pd.DataFrame(r["thresholds"]).set_index("threshold")
    t.columns = ["answered correctly", "answered wrongly", "refused", "nonsense rejected"]
    st.dataframe((t * 100).round(0).astype(int).astype(str) + "%", width="stretch")
    st.caption("Higher threshold = fewer wrong answers and more nonsense rejected, "
               "but more real questions refused. Pick the balance you want in "
               "`data/faqs.json` (confidence_threshold).")

# ---------------- Unanswered ----------------
st.subheader("What customers asked that the bot could not answer")
if r["clusters"]:
    for c in r["clusters"]:
        label = ", ".join(c["keywords"]) or "other"
        with st.expander(f"{c['count']} x  {label}"):
            for q in c["questions"]:
                st.write(f"- {q}")
            st.write(f"Closest existing intent: `{c['closest_intent']}` "
                     f"(confidence {c['avg_confidence']})")
else:
    st.write("Nothing logged yet. Chat with the bot and ask things it does not know.")