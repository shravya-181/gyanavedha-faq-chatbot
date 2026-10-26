"""
logger.py
Saves every question the bot could NOT answer into a CSV file.
The owner can read it to learn what customers want, and the Bot Doctor
(Step 5) will use it to suggest new intents.
"""

import csv
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = "data/unanswered.csv"
FIELDS = ["time", "question", "best_guess", "confidence"]


def log_unanswered(question: str, best_guess: str, confidence: float,
                   path: str = DEFAULT_PATH) -> None:
    """Add one unanswered question as a new row in the CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # make data/ if missing
    is_new_file = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new_file:
            writer.writeheader()                      # column names, once
        writer.writerow({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "best_guess": best_guess,
            "confidence": confidence,
        })


def read_unanswered(path: str = DEFAULT_PATH) -> list:
    """Return all logged questions as a list of dictionaries."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

        
# ---------------------------------------------------------------
# Feedback (the thumbs up / thumbs down buttons in the chat)
# ---------------------------------------------------------------
FEEDBACK_PATH = "data/feedback.csv"
FEEDBACK_FIELDS = ["time", "question", "answer", "rating"]


def log_feedback(question: str, answer: str, rating: str,
                 path: str = FEEDBACK_PATH) -> None:
    """Save one thumbs up ("up") or thumbs down ("down") for an answer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer[:200],
            "rating": rating,
        })


def read_feedback(path: str = FEEDBACK_PATH) -> list:
    """Return all saved feedback as a list of dictionaries."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))