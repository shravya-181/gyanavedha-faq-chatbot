"""Tests for the evaluation report and the dataset checks."""

import json

import pytest

from chatbot.evaluate import dataset_checks, evaluate


def make_faq(intents: dict) -> dict:
    return {
        "business_name": "Test College", "fallback_reply": "FALLBACK",
        "confidence_threshold": 0.35,
        "intents": [{"name": n, "answer": f"{n} answer", "examples": ex}
                    for n, ex in intents.items()],
    }


GOOD_FAQ = make_faq({
    "greeting": ["hi", "hello", "hey there", "good morning", "good evening", "namaste"],
    "fees": ["what is the fee structure", "how much is the tuition fee",
             "can i pay fees in installments", "fee details please",
             "what are the course charges", "how do i pay my fees online"],
    "hostel": ["is hostel available", "hostel facilities", "do you have a girls hostel",
               "what food is in the hostel mess", "hostel rules and timings",
               "accommodation for outstation students"],
    "exams": ["when are the exams", "exam timetable", "semester exam dates",
              "when will the hall ticket come", "how do i check my results",
              "internal exam schedule"],
})


@pytest.fixture
def paths(tmp_path):
    faq = tmp_path / "faqs.json"
    oos = tmp_path / "oos.json"
    faq.write_text(json.dumps(GOOD_FAQ), encoding="utf-8")
    oos.write_text(json.dumps(["tell me a joke", "what is the weather today"]), encoding="utf-8")
    return str(faq), str(oos)


def test_evaluate_returns_scores_between_0_and_1(paths):
    result = evaluate(paths[0], paths[1], save=False)
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1
    assert result["train_examples"] + result["test_examples"] == 24


def test_evaluate_covers_every_threshold_and_out_of_scope(paths):
    result = evaluate(paths[0], paths[1], save=False)
    assert len(result["threshold_sweep"]) == 10
    assert all("out_of_scope_rejected" in row for row in result["threshold_sweep"])
    assert result["out_of_scope_questions"] == 2


def test_higher_threshold_refuses_more(paths):
    sweep = evaluate(paths[0], paths[1], save=False)["threshold_sweep"]
    assert sweep[-1]["refused"] >= sweep[0]["refused"]


def test_evaluate_reports_error_when_data_is_too_small(tmp_path):
    tiny = tmp_path / "faqs.json"
    tiny.write_text(json.dumps(make_faq({"a": ["one"], "b": ["two"]})), encoding="utf-8")
    assert "error" in evaluate(str(tiny), save=False)


def test_dataset_checks_find_duplicates_and_conflicts(tmp_path):
    faq = make_faq({
        "fees": ["what is the fee", "what is the fee", "pay online"],
        "hostel": ["pay online", "hostel rooms"],
    })
    path = tmp_path / "faqs.json"
    path.write_text(json.dumps(faq), encoding="utf-8")
    warnings = " ".join(dataset_checks(str(path)))
    assert "repeats" in warnings
    assert "Conflict" in warnings
    assert "only 3 examples" in warnings


def test_dataset_checks_are_quiet_for_a_clean_file(paths):
    warnings = dataset_checks(paths[0])
    assert not any("Conflict" in w or "repeats" in w for w in warnings)