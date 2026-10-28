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


# ---------------- course awareness ----------------
COURSE_FAQ = dict(TEST_FAQ)
COURSE_FAQ["programs"] = [
    {"name": "BBA", "level": "UG", "aliases": ["bachelor of business administration"],
     "fee": "50,000", "duration": "3 years"},
    {"name": "MBA", "level": "PG", "aliases": [], "fee": "90,000", "duration": "2 years"},
]
COURSE_FAQ["not_offered"] = ["MTech", "BA"]
COURSE_FAQ["ambiguous"] = {"BSc": ["BSc Maths", "BSc Physics"]}
COURSE_FAQ["intents"] = [dict(i) for i in TEST_FAQ["intents"]]
for _i in COURSE_FAQ["intents"]:
    if _i["name"] == "fees":
        _i["program_answer"] = "{name} costs Rs {fee}"
        _i["level_answer"] = {"UG": "UG_FEES", "PG": "PG_FEES"}


@pytest.fixture
def course_bot(tmp_path):
    path = tmp_path / "faqs.json"
    path.write_text(json.dumps(COURSE_FAQ), encoding="utf-8")
    return Chatbot(str(path), str(tmp_path / "log.csv"))


def test_program_in_message_gives_program_answer(course_bot):
    assert "BBA costs Rs 50,000" in course_bot.reply("what is the BBA fee structure", {})["text"]


def test_bot_remembers_program(course_bot):
    context = {}
    course_bot.reply("what is the BBA fee structure", context)
    text = course_bot.reply("how much is the tuition fee", context)["text"]
    assert "BBA costs Rs 50,000" in text and "as you asked earlier" in text


def test_follow_up_what_about_other_program(course_bot):
    context = {}
    course_bot.reply("what is the BBA fee structure", context)
    assert "MBA costs Rs 90,000" in course_bot.reply("what about MBA", context)["text"]


def test_level_answer_for_ug(course_bot):
    assert course_bot.reply("fee details for ug students", {})["text"] == "UG_FEES"


def test_reset_forgets_program(course_bot):
    context = {}
    course_bot.reply("what is the BBA fee structure", context)
    course_bot.reply("reset", context)
    assert course_bot.reply("how much is the tuition fee", context)["text"] == "FEES_ANSWER"


def test_without_context_still_works(course_bot):
    assert course_bot.reply("how much is the tuition fee")["text"] == "FEES_ANSWER"


def test_three_dots_make_no_pieces():
    assert split_message("hello... fees") == ["hello", "fees"]


# ---------------- several programs in one message ----------------
def test_two_questions_two_programs(course_bot):
    text = course_bot.reply("what is the BBA fee structure? what about MBA?", {})["text"]
    assert "BBA costs Rs 50,000" in text and "MBA costs Rs 90,000" in text


def test_two_programs_in_one_question(course_bot):
    text = course_bot.reply("what is the fee structure for BBA and MBA", {})["text"]
    assert "BBA costs Rs 50,000" in text and "MBA costs Rs 90,000" in text


def test_programs_answered_in_the_order_asked(course_bot):
    text = course_bot.reply("what is the fee structure for MBA and BBA", {})["text"]
    assert text.index("MBA costs") < text.index("BBA costs")


def test_last_program_is_remembered(course_bot):
    context = {}
    course_bot.reply("what is the fee structure for BBA and MBA", context)
    assert context["program"] == "MBA"


# ---------------- names the college does not offer ----------------
def test_unknown_degree_is_reported_not_skipped(course_bot):
    text = course_bot.reply("what is the fee structure for BBA and MTech", {})["text"]
    assert "BBA costs Rs 50,000" in text
    assert "MTech" in text and "don't offer" in text


def test_only_unknown_degree_gets_a_clear_answer(course_bot):
    text = course_bot.reply("BA", {})["text"]
    assert "don't offer" in text and "BBA, MBA" in text


def test_ambiguous_name_asks_which_one(course_bot):
    text = course_bot.reply("what is the fee structure for BSc", {})["text"]
    assert "BSc Maths or BSc Physics" in text


def test_four_programs_in_a_row(course_bot):
    assert course_bot.reply("fee structure BBA MBA", {})["text"].count("costs Rs") == 2