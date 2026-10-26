"""
evaluate.py  -  an honest report card for the bot.

The examples in data/faqs.json are split in two:
  * 70% are used to TRAIN a copy of the model
  * 30% are hidden from it and used to TEST it
Questions that have nothing to do with the college (data/oos.json, "out of
scope") are used to check that the bot refuses what it should not answer.

Run it:   python -m chatbot.evaluate
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from chatbot.classifier import IntentClassifier
from chatbot.preprocess import clean_text

OOS_PATH = "data/oos.json"
REPORT_PATH = "reports/evaluation.json"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
MIN_EXAMPLES = 8


def load_oos(path: str = OOS_PATH) -> list:
    """Questions that must NOT be answered. One per line in a JSON list."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _max_probability(model, sentences: list) -> np.ndarray:
    """Confidence of the best intent for each sentence (0 if nothing is left after cleaning)."""
    cleaned = [clean_text(s) for s in sentences]
    result = np.zeros(len(cleaned))
    keep = [i for i, c in enumerate(cleaned) if c]
    if keep:
        result[keep] = model.predict_proba([cleaned[i] for i in keep]).max(axis=1)
    return result


def evaluate(faq_path: str = "data/faqs.json", oos_path: str = OOS_PATH,
             test_size: float = 0.3, seed: int = 42, save: bool = True) -> dict:
    with open(faq_path, encoding="utf-8") as f:
        data = json.load(f)

    X, y = [], []
    for intent in data["intents"]:
        for example in intent["examples"]:
            X.append(clean_text(example))
            y.append(intent["name"])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError as error:
        return {"error": f"Not enough examples to split: {error}"}

    # Train a fresh copy of the bot's model on the training part only.
    model = clone(IntentClassifier(faq_path).model).fit(X_train, y_train)
    classes = model.classes_
    proba = model.predict_proba(X_test)
    predicted = classes[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    y_test = np.array(y_test)

    # --- Overall and per-intent scores (ignoring the threshold) ---
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    per_intent = [
        {"intent": name, "precision": round(report[name]["precision"], 2),
         "recall": round(report[name]["recall"], 2), "f1": round(report[name]["f1-score"], 2),
         "test_examples": int(report[name]["support"])}
        for name in sorted(set(y_test))
    ]
    labels = sorted(set(y_test) | set(predicted))
    matrix = confusion_matrix(y_test, predicted, labels=labels).tolist()

    # --- Threshold sweep: real questions vs out-of-scope questions ---
    oos = load_oos(oos_path) if Path(oos_path).exists() else []
    oos_confidence = _max_probability(model, oos) if oos else np.array([])
    correct = predicted == y_test
    sweep = []
    for t in THRESHOLDS:
        answered = confidence >= t
        row = {
            "threshold": t,
            "answered_correctly": round(float((answered & correct).mean()), 2),
            "answered_wrongly": round(float((answered & ~correct).mean()), 2),
            "refused": round(float((~answered).mean()), 2),
        }
        if oos:
            row["out_of_scope_rejected"] = round(float((oos_confidence < t).mean()), 2)
            row["balanced_score"] = round(
                (row["answered_correctly"] + row["out_of_scope_rejected"]) / 2, 3)
        sweep.append(row)

    best = max(sweep, key=lambda r: r.get("balanced_score", r["answered_correctly"]))
    result = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "intents": len(set(y)),
        "train_examples": len(X_train),
        "test_examples": len(X_test),
        "out_of_scope_questions": len(oos),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 3),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 3),
        "weighted_f1": round(float(f1_score(y_test, predicted, average="weighted", zero_division=0)), 3),
        "current_threshold": data["confidence_threshold"],
        "best_threshold": best["threshold"],
        "per_intent": per_intent,
        "confusion_labels": labels,
        "confusion_matrix": matrix,
        "threshold_sweep": sweep,
    }
    if save:
        Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(REPORT_PATH).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def dataset_checks(faq_path: str = "data/faqs.json") -> list:
    """Warnings about the FAQ file itself (cheap checks that catch real problems)."""
    with open(faq_path, encoding="utf-8") as f:
        data = json.load(f)

    warnings = []
    counts = {i["name"]: len(i["examples"]) for i in data["intents"]}
    owners = {}      # cleaned example -> intents that contain it

    for intent in data["intents"]:
        seen = set()
        for example in intent["examples"]:
            cleaned = clean_text(example)
            if not cleaned:
                warnings.append(f"'{intent['name']}': example \"{example}\" is empty after cleaning.")
                continue
            if cleaned in seen:
                warnings.append(f"'{intent['name']}': \"{example}\" repeats another example.")
            seen.add(cleaned)
            owners.setdefault(cleaned, set()).add(intent["name"])

    for cleaned, names in owners.items():
        if len(names) > 1:
            warnings.append(f"Conflict: \"{cleaned}\" appears in {', '.join(sorted(names))}.")

    for name, n in counts.items():
        if n < MIN_EXAMPLES:
            warnings.append(f"'{name}' has only {n} examples (aim for {MIN_EXAMPLES}+).")

    if counts and max(counts.values()) > 2 * min(counts.values()):
        warnings.append("Imbalance: the biggest intent has more than twice the examples of the "
                        f"smallest ({Counter(counts).most_common(1)[0][0]} vs "
                        f"{min(counts, key=counts.get)}).")
    return warnings


if __name__ == "__main__":
    r = evaluate()
    if "error" in r:
        raise SystemExit(r["error"])
    print(f"Trained on {r['train_examples']} examples, tested on {r['test_examples']} unseen ones.")
    print(f"Accuracy {r['accuracy']:.0%} | macro-F1 {r['macro_f1']} | weighted-F1 {r['weighted_f1']}\n")
    print("Intent                 precision  recall  f1    tested")
    for p in r["per_intent"]:
        print(f"{p['intent']:22} {p['precision']:<10} {p['recall']:<7} {p['f1']:<5} {p['test_examples']}")
    print(f"\nThreshold sweep ({r['out_of_scope_questions']} out-of-scope questions):")
    print("threshold  answered-right  answered-wrong  refused  nonsense-rejected  balanced")
    for s in r["threshold_sweep"]:
        print(f"{s['threshold']:<10} {s['answered_correctly']:<15} {s['answered_wrongly']:<15} "
              f"{s['refused']:<8} {s.get('out_of_scope_rejected', '-'):<18} {s.get('balanced_score', '-')}")
    print(f"\nCurrent threshold: {r['current_threshold']}   Best by balanced score: {r['best_threshold']}")
    print("\nDataset checks:")
    for w in dataset_checks() or ["No problems found."]:
        print(" -", w)