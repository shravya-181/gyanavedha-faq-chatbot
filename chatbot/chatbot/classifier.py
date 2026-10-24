"""
classifier.py
Learns from data/faqs.json and decides which intent a message belongs to.

Pipeline:  raw text -> clean_text -> TF-IDF numbers -> logistic regression
Output:    the best intent, how confident we are, and the reply to give.
"""

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from chatbot.preprocess import clean_text


class IntentClassifier:
    def __init__(self, faq_path: str):
        # 1. Load the FAQ file.
        with open(faq_path, encoding="utf-8") as f:
            data = json.load(f)

        self.business_name = data["business_name"]
        self.fallback_reply = data["fallback_reply"]
        self.threshold = data["confidence_threshold"]

        # 2. Build the training data:
        #    X = cleaned example sentences, y = the intent each one belongs to.
        X, y = [], []
        self.answers = {}
        for intent in data["intents"]:
            self.answers[intent["name"]] = intent["answer"]
            for example in intent["examples"]:
                X.append(clean_text(example))
                y.append(intent["name"])

        # 3. Pipeline = text -> numbers (TF-IDF) -> intent probabilities
        #    (logistic regression).
        #    We build TWO kinds of TF-IDF features and combine them:
        #      - word features (1-2 words, e.g. "hostel fee")
        #      - character features (2-5 letters inside words), which still
        #        match when there are typos ("accomodation") or Hinglish words.
        #    C=50 lets the model be more decisive; a small dataset gives very
        #    timid probabilities with the default C=1.
        features = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2))),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))),
        ])
        self.model = Pipeline([
            ("tfidf", features),
            ("clf", LogisticRegression(C=50, max_iter=2000)),
        ])

        # 4. Learn from the examples.
        self.model.fit(X, y)

    def predict(self, message: str, top_k: int = 3) -> dict:
        """Return the best intent, its confidence, the reply and the top matches."""
        cleaned = clean_text(message)

        # A message with no useful words (e.g. "???") cannot be understood.
        if not cleaned:
            return self._fallback(cleaned, [])

        # Probability for every intent, e.g. {"fees": 0.62, "hostel": 0.11, ...}
        probabilities = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_

        # Sort intents from most to least likely.
        ranked = sorted(zip(classes, probabilities), key=lambda p: p[1], reverse=True)
        top = [(str(name), round(float(prob), 3)) for name, prob in ranked[:top_k]]

        best_intent, best_confidence = str(ranked[0][0]), ranked[0][1]
        best_confidence = float(best_confidence)

        # Below the threshold, we are not sure enough, so we use the fallback.
        if best_confidence < self.threshold:
            return self._fallback(cleaned, top)

        return {
            "intent": best_intent,
            "confidence": round(best_confidence, 3),
            "answer": self.answers[best_intent],
            "is_fallback": False,
            "top_matches": top,
            "cleaned": cleaned,
        }

    def _fallback(self, cleaned: str, top: list) -> dict:
        return {
            "intent": None,
            "confidence": top[0][1] if top else 0.0,
            "answer": self.fallback_reply,
            "is_fallback": True,
            "top_matches": top,
            "cleaned": cleaned,
        }


if __name__ == "__main__":
    bot = IntentClassifier("data/faqs.json")
    questions = [
        "hi there",
        "thanks a lot",
        "how much do i have to pay per year",
        "is accomodation available for girls",
        "when will my results come out",
        "what is the capital of France",
        "???",
        "what about placments?"
        "highest salary package for pg?"
        "does the fees must be paid in one shot?"
        "is there any annual fest?"
        "what is the capital of india?"
    ]   
    for q in questions:
        r = bot.predict(q)
        label = "FALLBACK" if r["is_fallback"] else r["intent"]
        print(f"{q!r:42} -> {label:15} conf={r['confidence']}  top={r['top_matches']}")