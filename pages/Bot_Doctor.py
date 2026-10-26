"""
pages/Bot_Doctor.py  -  the Bot Doctor page.
Streamlit shows any file inside a `pages` folder as an extra page in the sidebar.
"""

import pandas as pd
import streamlit as st

from chatbot.doctor import diagnose
from chatbot.evaluate import dataset_checks, evaluate
from chatbot.logger import read_feedback

st.set_page_config(page_title="Bot Doctor", page_icon="🩺", layout="wide")


def staff_only() -> None:
    """Ask for the staff password. The password lives in Streamlit secrets
    (key: STAFF_PASSWORD), never in the code. If no password is set (for
    example when running on your own computer), the page stays open."""
    try:
        password = st.secrets.get("STAFF_PASSWORD", "")
    except Exception:
        password = ""
    if not password or st.session_state.get("staff_ok"):
        return
    st.title("🩺 Bot Doctor")
    st.info("This page is for college staff. Please enter the staff password.")
    typed = st.text_input("Staff password", type="password")
    if typed:
        if typed == password:
            st.session_state.staff_ok = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


staff_only()
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

# ---------------- Feedback ----------------
st.subheader("What customers thought of the answers")
feedback = read_feedback()
if feedback:
    ups = sum(f["rating"] == "up" for f in feedback)
    downs = len(feedback) - ups
    st.write(f"{ups} thumbs up, {downs} thumbs down")
    bad = [f for f in feedback if f["rating"] == "down"]
    if bad:
        st.write("Answers marked as unhelpful (fix these first):")
        st.dataframe(pd.DataFrame(bad)[["question", "answer"]], hide_index=True, width="stretch")
else:
    st.write("No feedback yet. Use the thumbs buttons under the bot's answers.")

# ---------------- Held-out evaluation ----------------
st.subheader("Report card on hidden test sentences")
st.caption("30% of your examples are hidden from the bot while it learns, then used as a test. "
           "Questions from data/oos.json (things a college bot should refuse) check the threshold.")
if st.button("Run evaluation") or "evaluation" in st.session_state:
    if "evaluation" not in st.session_state:
        with st.spinner("Testing the bot..."):
            st.session_state.evaluation = evaluate(save=False)
    ev = st.session_state.evaluation
    if "error" in ev:
        st.warning(ev["error"])
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Accuracy", f"{int(ev['accuracy'] * 100)}%")
        e2.metric("Macro F1", ev["macro_f1"])
        e3.metric("Tested on", f"{ev['test_examples']} sentences")
        e4.metric("Best threshold", ev["best_threshold"])

        st.write("**Threshold table** (what happens at each cut-off)")
        sweep = pd.DataFrame(ev["threshold_sweep"]).set_index("threshold")
        sweep.columns = ["answered correctly", "answered wrongly", "refused",
                         "nonsense rejected", "balanced score"][:len(sweep.columns)]
        st.dataframe(sweep, width="stretch")
        st.caption(f"Your current threshold is {ev['current_threshold']}. "
                   "Balanced score = average of 'answered correctly' and 'nonsense rejected'. "
                   "A higher cut-off refuses more nonsense but also more real questions, "
                   "so choose the balance you want.")

        st.write("**Per intent**")
        st.dataframe(pd.DataFrame(ev["per_intent"]).set_index("intent"), width="stretch")

        st.write("**Confusion matrix** (rows = what the customer meant, columns = what the bot chose)")
        st.dataframe(pd.DataFrame(ev["confusion_matrix"], index=ev["confusion_labels"],
                                  columns=ev["confusion_labels"]), width="stretch")

# ---------------- Dataset checks ----------------
st.subheader("Checks on faqs.json")
problems = dataset_checks()
if problems:
    for problem in problems:
        st.write(f"- {problem}")
else:
    st.write("No problems found.")
