"""
bot.py
The "brain" the chat page talks to. It wraps the classifier and adds:
  1. multi-question handling (one message can contain several questions)
  2. logging of every question the bot could not answer
  3. course awareness: it notices programs (BBA, MBA...) and levels (UG / PG),
     remembers them during the chat, and gives answers for that program
"""

import re

from chatbot.classifier import IntentClassifier
from chatbot.logger import DEFAULT_PATH, log_unanswered
from chatbot.preprocess import clean_text

# Words that mean UG or PG. (Program names come from faqs.json, not from here.)
LEVEL_WORDS = {
    "UG": ["ug", "undergraduate", "under graduate", "after 12th", "after 12"],
    "PG": ["pg", "postgraduate", "post graduate", "masters", "after graduation"],
}
RESET_WORDS = {"reset", "forget", "forget program", "start over", "clear", "new question"}
# "what about MBA", "and for BCA", "same for MCom" ... = same topic, new program
FOLLOW_UP = re.compile(r"^(what|how) about\b|^and\b|^same for\b|^for\b|^ok (and|for)\b")


def split_message(message: str) -> list:
    """Split one message into separate questions at ? . ! ; and new lines.

    "fees? hostel. any bus" -> ["fees", "hostel", "any bus"]
    A dot inside a word ("B.Com") does not split.
    """
    pieces = re.split(r"[?!;\n]+|\.+(?=\s|$)", message)
    return [p.strip() for p in pieces if p.strip()]


def normalise(text: str) -> str:
    """Lowercase and remove dots/apostrophes so "B.Com" matches "bcom"."""
    text = re.sub(r"(?<=\w)\.(?=\w)", "", text.lower())
    return text.replace("'", "").replace("’", "")


class _Blank(dict):
    """For str.format_map: a field the program does not have becomes empty."""

    def __missing__(self, key):
        return ""


class Chatbot:
    def __init__(self, faq_path: str = "data/faqs.json",
                 log_path: str = DEFAULT_PATH):
        self.classifier = IntentClassifier(faq_path)
        self.log_path = log_path
        self.programs = {p["name"]: p for p in self.classifier.programs}

    # ---------------- course awareness ----------------
    def find_program(self, text: str):
        """Return the program mentioned in the text (longest alias wins), or None."""
        t = normalise(text)
        best = None
        for prog in self.programs.values():
            for alias in [prog["name"], *prog.get("aliases", [])]:
                a = normalise(alias)
                if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", t):
                    if best is None or len(a) > best[0]:
                        best = (len(a), prog)
        return best[1] if best else None

    def find_programs(self, text: str) -> list:
        """Return EVERY program mentioned, in the order they appear.

        "BBA and MBA fees" -> [BBA, MBA].  If one name sits inside another
        ("msc" inside "msc computer science") the longer name wins.
        """
        t = normalise(text)
        hits = []                                   # (start, end, program)
        for prog in self.programs.values():
            for alias in [prog["name"], *prog.get("aliases", [])]:
                a = normalise(alias)
                for m in re.finditer(rf"(?<!\w){re.escape(a)}(?!\w)", t):
                    hits.append((m.start(), m.end(), prog))
        hits.sort(key=lambda h: h[0] - h[1])        # longest first
        taken, chosen = [], []
        for start, end, prog in hits:
            if any(start < e and s < end for s, e in taken):
                continue                            # overlaps a longer match
            taken.append((start, end))
            chosen.append((start, prog))
        found = []
        for _, prog in sorted(chosen, key=lambda c: c[0]):
            if prog not in found:
                found.append(prog)
        return found

    def strip_programs(self, text: str) -> str:
        """The question with every program name removed: "fees of BBA and MBA" -> "fees of and".
        Program names are strong words, so they can hide the real topic."""
        aliases = sorted({normalise(a) for p in self.programs.values()
                          for a in [p["name"], *p.get("aliases", [])]}, key=len, reverse=True)
        if not aliases:
            return text
        pattern = "|".join(re.escape(a) for a in aliases)
        return re.sub(rf"(?<!\w)(?:{pattern})(?!\w)", " ", normalise(text))

    @staticmethod
    def find_level(text: str):
        t = normalise(text)
        for level, words in LEVEL_WORDS.items():
            if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t) for w in words):
                return level
        return None

    def _answer_for(self, intent: str, context: dict, from_memory: bool) -> str:
        """Pick the program answer, the UG/PG answer, or the general answer."""
        extra = self.classifier.extras.get(intent, {})
        prog = self.programs.get(context.get("program"))
        if prog and extra.get("program_answer"):
            text = extra["program_answer"].format_map(_Blank(prog))
            if from_memory:
                text += (f"\n\n*Showing **{prog['name']}**, as you asked earlier. "
                         "Type **reset** to see general answers.*")
            return text
        level = context.get("level")
        if level and extra.get("level_answer", {}).get(level):
            return extra["level_answer"][level]
        return self.classifier.answers[intent]

    # ---------------- replying ----------------
    def reply(self, message: str, context: dict = None) -> dict:
        """Answer a message.

        `context` is a dict the chat page keeps between messages. The bot reads
        and updates it: {"program": "BBA", "level": "UG", "last_intent": "fees"}.
        """
        if context is None:
            context = {}

        if normalise(message).strip(" .!") in RESET_WORDS:
            context.clear()
            return {"text": "Done. I've cleared what I remembered, so my answers "
                            "are general again. What would you like to know?",
                    "details": [], "unanswered": []}

        pieces = []
        for piece in split_message(message) or [message]:
            if len(self.find_programs(piece)) > 1:   # "BBA and MBA fees": keep whole
                pieces.append(piece)
            else:
                pieces.extend(self._split_on_and(piece))

        details = []          # one prediction per piece
        answers = []          # replies we are confident about
        unanswered = []       # pieces we could not answer
        seen = set()          # so two "placement" questions give one answer

        for piece in pieces:
            found = self.find_programs(piece)
            program = found[-1] if found else None   # the last one is remembered
            level = self.find_level(piece)
            if program:
                context["program"], context["level"] = program["name"], program["level"]
            elif level:
                context["program"], context["level"] = None, level

            result = self.classifier.predict(piece)
            result["question"] = piece
            has_template = lambda name: "program_answer" in self.classifier.extras.get(name, {})  # noqa: E731

            # A program name can hide the real topic ("fees of BBA, MBA and MCom"
            # looks like "courses"). Ask again without the names: if that clearly
            # says fees or admission, trust it.
            if found and result["intent"] in (None, "courses"):
                rest = self.strip_programs(piece)
                if clean_text(rest):
                    alt = self.classifier.predict(rest)
                    if (not alt["is_fallback"] and alt["intent"] != "courses"
                            and has_template(alt["intent"])):
                        result.update(intent=alt["intent"], confidence=alt["confidence"],
                                      is_fallback=False, answer=alt["answer"])
            intent = result["intent"]

            # "what about MBA?" -> repeat the last topic for the new program.
            if program and (result["is_fallback"]
                            or FOLLOW_UP.match(normalise(piece).strip())):
                last = context.get("last_intent")
                intent = last if last and has_template(last) else "courses"
                if has_template(intent):
                    result.update(intent=intent, is_fallback=False)
                    result["confidence"] = max(result["confidence"], self.classifier.threshold)

            result["program"] = context.get("program")
            details.append(result)

            if result["is_fallback"] or intent is None:
                # Skip meaningless leftovers such as "ok" when the message has
                # several parts, but always log a single-part message.
                if result["cleaned"] or len(pieces) == 1:
                    unanswered.append(piece)
                    best_guess = result["top_matches"][0][0] if result["top_matches"] else ""
                    log_unanswered(piece, best_guess, result["confidence"], self.log_path)
                continue

            if has_template(intent) or intent in self.classifier.extras and \
                    "level_answer" in self.classifier.extras[intent]:
                context["last_intent"] = intent
            if len(found) > 1 and has_template(intent):
                # "BBA and MBA fees": one answer for each program mentioned (max 4)
                for prog in found[:4]:
                    key = (intent, prog["name"], prog["level"])
                    if key not in seen:
                        seen.add(key)
                        answers.append(self._answer_for(
                            intent, {"program": prog["name"], "level": prog["level"]}, False))
                if len(found) > 4:
                    answers.append("*I showed the first 4 programs. Ask about the others "
                                   "separately and I'll answer them too.*")
                continue
            key = (intent, context.get("program"), context.get("level"))
            if key not in seen:
                seen.add(key)
                from_memory = bool(context.get("program")) and not program
                answers.append(self._answer_for(intent, context, from_memory))

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
        text = "\n\n---\n\n".join(answers)
        if unanswered:
            quoted = "; ".join(f'"{q}"' for q in unanswered)
            text += (f"\n\nI couldn't answer: {quoted}. "
                     "I've noted it so the team can teach me.")
        return text
