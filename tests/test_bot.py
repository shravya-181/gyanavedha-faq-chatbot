"""
Automated tests. Run them with:   python -m pytest -v
Each test checks one small promise the bot makes.
"""

import json

import pytest

from chatbot.bot import Chatbot, split_message
from chatbot.classifier import IntentClassifier
from chatbot.logger import read_unanswered
from chatbot.preprocess import clean_text

# A tiny FAQ file used only for testing, so the tests do not break
# when you edit your real data/faqs.json.
TEST_FAQ = {
    "business_name": "Test College",
    "fallback_reply": "FALLBACK",
    "confidence_threshold": 0.35,
    "intents": [
        {"name": "greeting", "answer": "GREETING_ANSWER", "examples": [
            "hi", "hello", "hey there", "good morning", "good evening", "namaste"]},
        {"name": "fees", "answer": "FEES_ANSWER", "examples": [
            "what is the fee structure", "how much is the tuition fee",
            "can i pay fees in installments", "fee details please",
            "what are the course charges", "how do i pay my fees online"]},
        {"name": "hostel", "answer": "HOSTEL_ANSWER", "examples": [
            "is hostel available", "hostel facilities", "do you have a girls hostel",
            "what food is in the hostel mess", "hostel rules and timings",
            "accommodation for outstation students"]},
        {"name": "exams", "answer": "EXAMS_ANSWER", "examples": [
            "when are the exams", "exam timetable", "semester exam dates",
            "when will the hall ticket come", "how do i check my results",
            "internal exam schedule"]},
        {"name": "transport", "answer": "BUS_ANSWER", "examples": [
            "is there a college bus", "bus timings", "which routes does the bus cover",
            "bus pass details", "transport facility for day scholars",
            "how do i reach the college by bus"]},
    ],
}


@pytest.fixture
def faq_path(tmp_path):
    path = tmp_path / "faqs.json"
    path.write_text(json.dumps(TEST_FAQ), encoding="utf-8")
    return str(path)


@pytest.fixture
def bot(faq_path, tmp_path):
    return Chatbot(faq_path, str(tmp_path / "unanswered.csv"))


# ---------- preprocess ----------
def test_clean_lowercases_and_removes_punctuation():
    assert clean_text("HELLO!!!") == "hello"


def test_clean_removes_filler_words():
    assert clean_text("the fees of the college") == "fee college"


def test_clean_keeps_negations_and_question_words():
    cleaned = clean_text("When can't I pay?")
    assert "when" in cleaned.split()
    assert "cant" in cleaned.split()


def test_clean_lemmatizes_plurals():
    assert clean_text("fees") == "fee"


def test_clean_empty_text():
    assert clean_text("???") == ""


# ---------- classifier ----------
def test_classifier_recognises_greeting(faq_path):
    result = IntentClassifier(faq_path).predict("hello")
    assert result["intent"] == "greeting"
    assert result["answer"] == "GREETING_ANSWER"


def test_classifier_recognises_new_sentence(faq_path):
    result = IntentClassifier(faq_path).predict("when will the exam results come")
    assert result["intent"] == "exams"


def test_gibberish_uses_fallback(faq_path):
    result = IntentClassifier(faq_path).predict("xyzzy qwerty zzzz")
    assert result["is_fallback"] is True
    assert result["answer"] == "FALLBACK"


def test_empty_message_uses_fallback(faq_path):
    result = IntentClassifier(faq_path).predict("???")
    assert result["is_fallback"] is True
    assert result["confidence"] == 0.0


def test_confidence_is_a_probability(faq_path):
    result = IntentClassifier(faq_path).predict("hostel fee")
    assert 0.0 <= result["confidence"] <= 1.0


# ---------- splitting ----------
def test_split_on_question_marks():
    assert split_message("fees? hostel? bus") == ["fees", "hostel", "bus"]


def test_split_ignores_empty_pieces():
    assert split_message("?? ! ...") == []


# ---------- full bot ----------
def test_bot_answers_two_topics_in_one_message(bot):
    text = bot.reply("fees and hostel")["text"]
    assert "FEES_ANSWER" in text and "HOSTEL_ANSWER" in text


def test_bot_does_not_split_unknown_and_phrases(bot):
    pieces = [d["question"] for d in bot.reply("terms and conditions")["details"]]
    assert pieces == ["terms and conditions"]


def test_bot_gives_one_answer_for_repeated_topic(bot):
    text = bot.reply("hostel facilities? is hostel available?")["text"]
    assert text.count("HOSTEL_ANSWER") == 1


def test_unanswered_question_is_logged(bot, tmp_path):
    reply = bot.reply("xyzzy qwerty zzzz")
    assert reply["text"] == "FALLBACK"
    rows = read_unanswered(str(tmp_path / "unanswered.csv"))
    assert len(rows) == 1 and rows[0]["question"] == "xyzzy qwerty zzzz"


def test_answered_question_is_not_logged(bot, tmp_path):
    bot.reply("hello")
    assert read_unanswered(str(tmp_path / "unanswered.csv")) == []


def test_partly_answered_message_logs_only_the_unknown_part(bot, tmp_path):
    reply = bot.reply("hello? xyzzy qwerty zzzz")
    assert "GREETING_ANSWER" in reply["text"]
    assert "couldn't answer" in reply["text"]
    assert len(read_unanswered(str(tmp_path / "unanswered.csv"))) == 1


# ---------- your real FAQ file ----------
def test_real_faq_file_is_well_formed():
    """Checks data/faqs.json (your real data) for common mistakes."""
    with open("data/faqs.json", encoding="utf-8") as f:
        data = json.load(f)
    assert 0 < data["confidence_threshold"] < 1
    names = [i["name"] for i in data["intents"]]
    assert len(names) == len(set(names)), "duplicate intent names"
    for intent in data["intents"]:
        assert len(intent["examples"]) >= 5, f"{intent['name']} needs more examples"
        assert intent["answer"].strip(), f"{intent['name']} has an empty answer"