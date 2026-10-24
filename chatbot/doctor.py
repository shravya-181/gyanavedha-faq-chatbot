"""
doctor.py  -  the Bot Doctor.

Looks at the bot's FAQ file and its unanswered-question log and reports:
  1. how accurate the bot is on questions it has not memorised (cross-validation)
  2. which intents get confused with each other
  3. which intents are weak (too few examples / low accuracy)
  4. how the confidence threshold behaves
  5. what customers asked that the bot could not answer (grouped)
  6. plain-English recommendations
"""

import json
from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from chatbot.classifier import IntentClassifier
from chatbot.logger import DEFAULT_PATH, read_unanswered
from chatbot.preprocess import clean_text

MIN_EXAMPLES = 8          # fewer examples than this = "thin" intent
MIN_RECALL = 0.70         # below this accuracy an intent is "weak"
OVERLAP_LIMIT = 0.15      # similarity above this = intents look alike
                          # (small FAQ files give low scores, so 0.15 is already notable)
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# Words that start questions but say nothing about the topic.
QUESTION_WORDS = {"what", "when", "where", "which", "who", "how", "why",
                  "not", "no", "nor", "cant", "dont", "wont", "cannot"}

# Questions that have NOTHING to do with a college. A good bot must refuse them.
OUT_OF_SCOPE = [
    "what is the capital of France", "who won the cricket match",
    "tell me a joke", "how to cook biryani", "what is the weather today",
    "best movie of this year", "how do I fix my laptop",
    "what is the price of gold", "recommend a good restaurant",
    "who is the prime minister",
]


def _load_training(faq_path):
    """Return (X, y): cleaned example sentences and the intent of each."""
    with open(faq_path, encoding="utf-8") as f:
        data = json.load(f)
    X, y = [], []
    for intent in data["intents"]:
        for example in intent["examples"]:
            X.append(clean_text(example))
            y.append(intent["name"])
    return X, y


def _cross_validate(model, X, y):
    """Predict every example using a model that has NOT seen it.

    The data is cut into folds. For each fold, we train on the other folds and
    predict this one. This is an honest estimate of how the bot does on new
    sentences (testing on the training sentences would be cheating).
    """
    smallest = min(Counter(y).values())
    folds = min(5, smallest)
    if folds < 2:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    predicted = classes[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    return predicted, confidence


def _intent_overlaps(X, y):
    """How similar are two intents? Compare their combined example text."""
    names = sorted(set(y))
    docs = [" ".join(x for x, label in zip(X, y) if label == n) for n in names]
    matrix = TfidfVectorizer().fit_transform(docs)
    sims = cosine_similarity(matrix)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sims[i, j] >= OVERLAP_LIMIT:
                pairs.append({"a": names[i], "b": names[j],
                              "similarity": round(float(sims[i, j]), 2)})
    return sorted(pairs, key=lambda p: p["similarity"], reverse=True)


def _cluster_unanswered(rows):
    """Group similar unanswered questions so the owner sees what is asked most."""
    if not rows:
        return []
    questions = [r["question"] for r in rows]
    cleaned = [clean_text(q) or q.lower() for q in questions]
    vectors = TfidfVectorizer().fit_transform(cleaned)
    sims = cosine_similarity(vectors)

    clusters = []                       # each cluster = list of row indexes
    for i in range(len(rows)):
        for cluster in clusters:
            if np.mean([sims[i, j] for j in cluster]) >= 0.35:
                cluster.append(i)
                break
        else:
            clusters.append([i])

    result = []
    for cluster in clusters:
        members = [rows[i] for i in cluster]
        words = Counter(w for i in cluster for w in cleaned[i].split()
                        if w not in QUESTION_WORDS)
        guesses = Counter(m["best_guess"] for m in members if m["best_guess"])
        closest = guesses.most_common(1)[0][0] if guesses else ""
        avg_conf = float(np.mean([float(m["confidence"]) for m in members]))
        result.append({
            "count": len(members),
            "questions": sorted({m["question"] for m in members}),
            "keywords": [w for w, _ in words.most_common(3)],
            "closest_intent": closest,
            "avg_confidence": round(avg_conf, 2),
        })
    return sorted(result, key=lambda c: c["count"], reverse=True)


def diagnose(faq_path="data/faqs.json", log_path=DEFAULT_PATH):
    bot = IntentClassifier(faq_path)
    X, y = _load_training(faq_path)
    counts = Counter(y)

    report = {
        "n_intents": len(counts),
        "n_examples": len(X),
        "current_threshold": bot.threshold,
        "accuracy": None, "per_intent": [], "confusions": [],
        "overlaps": _intent_overlaps(X, y), "weak": [],
        "thresholds": [], "clusters": _cluster_unanswered(read_unanswered(log_path)),
        "recommendations": [],
    }

    cv = _cross_validate(bot.model, X, y)
    recall = {}
    confidence = None
    if cv is not None:
        predicted, confidence = cv
        truth = np.array(y)
        report["accuracy"] = round(float((predicted == truth).mean()), 3)

        for name in sorted(counts):
            mask = truth == name
            recall[name] = float((predicted[mask] == name).mean())
            report["per_intent"].append({
                "intent": name, "examples": counts[name],
                "accuracy": round(recall[name], 2)})

        wrong = Counter((str(t), str(p)) for t, p in zip(truth, predicted) if t != p)
        report["confusions"] = [
            {"intent": t, "mistaken_for": p, "times": n}
            for (t, p), n in wrong.most_common()]

        # Threshold table: in-scope (cross-validated) vs out-of-scope questions.
        correct = predicted == truth
        oos_conf = np.array([
            bot.model.predict_proba([clean_text(q)])[0].max() for q in OUT_OF_SCOPE])
        for t in THRESHOLDS:
            report["thresholds"].append({
                "threshold": t,
                "answered_correctly": round(float(((confidence >= t) & correct).mean()), 2),
                "answered_wrongly": round(float(((confidence >= t) & ~correct).mean()), 2),
                "refused": round(float((confidence < t).mean()), 2),
                "nonsense_rejected": round(float((oos_conf < t).mean()), 2),
            })

    # Weak intents
    for name in sorted(counts):
        reasons = []
        if counts[name] < MIN_EXAMPLES:
            reasons.append(f"only {counts[name]} examples (aim for {MIN_EXAMPLES}+)")
        if name in recall and recall[name] < MIN_RECALL:
            reasons.append(f"correct only {int(recall[name] * 100)}% of the time on new sentences")
        if reasons:
            report["weak"].append({"intent": name, "reasons": reasons})

    report["recommendations"] = _recommend(report)
    return report


def _recommend(r):
    tips = []
    for c in r["confusions"][:3]:
        tips.append(f"'{c['intent']}' was mistaken for '{c['mistaken_for']}' "
                    f"{c['times']} time(s). Add examples to '{c['intent']}' that use "
                    f"words only that topic would use.")
    for o in r["overlaps"][:3]:
        tips.append(f"'{o['a']}' and '{o['b']}' share much of the same wording "
                    f"(similarity {o['similarity']}). Rewrite some examples so they differ, "
                    f"or merge the two intents.")
    for w in r["weak"][:3]:
        tips.append(f"'{w['intent']}': {'; '.join(w['reasons'])}. Write more varied examples.")
    for c in r["clusters"][:3]:
        label = ", ".join(c["keywords"]) or "this topic"
        tips.append(f"{c['count']} customer(s) asked about '{label}' and the bot could not "
                    f"answer (e.g. \"{c['questions'][0]}\"). Create a new intent, or add these "
                    f"as examples to '{c['closest_intent']}' if it is the same topic.")
    if not tips:
        tips.append("No problems found. Keep adding real customer questions to stay sharp.")
    return tips


if __name__ == "__main__":
    rep = diagnose()
    print(f"{rep['n_intents']} intents, {rep['n_examples']} examples, "
          f"accuracy on unseen sentences: {rep['accuracy']}")
    print("\nPer intent:", [(p['intent'], p['accuracy']) for p in rep['per_intent']])
    print("\nConfusions:", rep["confusions"])
    print("\nOverlaps:", rep["overlaps"])
    print("\nWeak:", rep["weak"])
    print("\nThresholds:")
    for t in rep["thresholds"]:
        print("  ", t)
    print("\nUnanswered groups:", rep["clusters"])
    print("\nRecommendations:")
    for tip in rep["recommendations"]:
        print(" -", tip)