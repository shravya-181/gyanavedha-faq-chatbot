"""
bot.py
The "brain" the chat page talks to. It wraps the classifier and adds:
  1. multi-question handling (one message can contain several questions)
  2. logging of every question the bot could not answer
"""

import re

from chatbot.classifier import IntentClassifier
from chatbot.logger import DEFAULT_PATH, log_unanswered


def split_message(message: str) -> list:
    """Split one message into separate questions at ? . ! ; and new lines.

    "fees? hostel. any bus" -> ["fees", "hostel", "any bus"]
    """
    pieces = re.split(r"[?.!;\n]+", message)
    return [p.strip() for p in pieces if p.strip()]


class Chatbot:
    def __init__(self, faq_path: str = "data/faqs.json",
                 log_path: str = DEFAULT_PATH):
        self.classifier = IntentClassifier(faq_path)
        self.log_path = log_path

    def reply(self, message: str) -> dict:
        """Answer a message. Returns the reply text plus details for debugging."""
        pieces = []
        for piece in split_message(message) or [message]:
            pieces.extend(self._split_on_and(piece))

        details = []          # one prediction per piece
        answers = []          # replies we are confident about
        unanswered = []       # pieces we could not answer
        seen_intents = set()  # so two "placement" questions give one answer

        for piece in pieces:
            result = self.classifier.predict(piece)
            result["question"] = piece
            details.append(result)

            if result["is_fallback"]:
                # Skip meaningless leftovers such as "ok" when the message has
                # several parts, but always log a single-part message.
                if result["cleaned"] or len(pieces) == 1:
                    unanswered.append(piece)
                    best_guess = result["top_matches"][0][0] if result["top_matches"] else ""
                    log_unanswered(piece, best_guess, result["confidence"], self.log_path)
            elif result["intent"] not in seen_intents:
                seen_intents.add(result["intent"])
                answers.append(result["answer"])

        return {
            "text": self._compose(answers, unanswered),
            "details": details,
            "unanswered": unanswered,
        }

    def _split_on_and(self, piece: str) -> list:
        """Split "fees and hostel" into two questions, but only when it is safe.

        We split on the word "and", ask the classifier about each half, and
        keep the split only if EVERY half is understood and the halves are
        DIFFERENT intents. Otherwise ("terms and conditions") keep it whole.
        """
        parts = [p.strip() for p in re.split(r"\s+and\s+", piece, flags=re.IGNORECASE)
                 if p.strip()]
        if len(parts) < 2:
            return [piece]

        results = [self.classifier.predict(p) for p in parts]
        all_understood = all(not r["is_fallback"] for r in results)
        all_different = len({r["intent"] for r in results}) == len(results)
        return parts if (all_understood and all_different) else [piece]

    def _compose(self, answers: list, unanswered: list) -> str:
        if not answers:
            return self.classifier.fallback_reply
        text = "\n\n".join(answers)
        if unanswered:
            quoted = "; ".join(f'"{q}"' for q in unanswered)
            text += (f"\n\nI couldn't answer: {quoted}. "
                     "I've noted it so the team can teach me.")
        return text