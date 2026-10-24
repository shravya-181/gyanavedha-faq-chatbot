"""
preprocess.py
Turns a raw customer message into a clean, standard form
that the intent classifier can learn from.

Example:  "Where is my FEES receipt??"  ->  "where fee receipt"
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    """Download NLTK's word lists if they are missing.

    On your laptop they are already there. On a fresh cloud server (such as
    Streamlit Cloud) they are not, so we fetch them the first time.
    """
    needed = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
    ]
    for path, package in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

# ---------------------------------------------------------------
# One-time setup (done when the file is imported, not on every call)
# ---------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()

# Words that NLTK calls "stop words" but that CHANGE the meaning of a
# question, so we keep them. Example: "can't pay" must not become "pay".
_KEEP_WORDS = {
    "not", "no", "nor", "cant", "dont", "wont", "cannot",   # negations
    "what", "when", "where", "which", "who", "how", "why",  # question words
}

# All filler words to drop = NLTK's list minus the words we keep.
_STOP_WORDS = set(stopwords.words("english")) - _KEEP_WORDS


def clean_text(text: str) -> str:
    """Return a cleaned version of `text` as one string of words."""
    # 1. Lowercase, so "FEES" and "fees" are the same word.
    text = text.lower()

    # 2. Remove apostrophes first, so "can't" becomes "cant" (one word)
    #    instead of splitting into "can" and "t".
    text = text.replace("'", "")

    # 3. Replace everything that is not a letter, digit or space with a
    #    space. This removes punctuation like ? ! , . and symbols.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Split the sentence into a list of words (tokens).
    tokens = word_tokenize(text)

    # 5. Drop filler words such as "is", "the", "my".
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    # 6. Lemmatize: bring each word to its base form.
    #    First as a noun ("fees" -> "fee"), then as a verb ("joining" -> "join").
    tokens = [_lemmatizer.lemmatize(t, pos="n") for t in tokens]
    tokens = [_lemmatizer.lemmatize(t, pos="v") for t in tokens]

    # 7. Join the words back into a single string.
    return " ".join(tokens)


if __name__ == "__main__":
    tests = [
        "Where is my FEES receipt??",
        "What are the hostel charges for PG students?",
        "hostle milega kya",
        "I can't pay the fees online!",
        "When does the MBA admission process start?",
        "Are you joining the placements drive?",
    ]
    for t in tests:
        print(f"{t!r:55} -> {clean_text(t)!r}")