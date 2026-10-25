# Gyanavedha FAQ Chatbot

A college help-desk chatbot that answers student and parent questions about admissions, fees, hostel, courses, exams, scholarships, placements and more. It runs in the browser, learns from a plain **JSON FAQ file**, and comes with a **Bot Doctor** page that tells the owner how to make the bot better.

> **Live demo:** https://YOUR-APP-NAME.streamlit.app
>
> Gyanavedha Institute is a **fictional demo college**. All fees, dates and contact details are sample data.

Built for the 2-Month Artificial Intelligence Internship (Project 6: Chatbot for Customer Service).

---

## What makes it different

- **Plug-and-play.** The bot's knowledge lives in `data/faqs.json`, not in the code. Any business can edit one file and get its own chatbot, with no retraining by hand.
- **Bot Doctor.** A second page that checks the bot's own quality: accuracy on sentences it has never seen, which topics get mixed up, which intents are weak, and what customers asked that the bot could not answer.
- **Several questions in one message.** "fees and hostel? is there an annual fest?" is split and answered part by part. Unknown parts are logged instead of guessed.
- **Honest fallback.** If the bot is not confident enough, it says so instead of giving a wrong answer, and saves the question.
- **Typo tolerant.** Character-level features help with spelling mistakes and Hinglish words such as "hostle" or "fee kitna".
- **Customer feedback.** 👍 / 👎 under every answer, shown in the Bot Doctor.

## How it works

```
message -> split into questions -> clean the text -> TF-IDF features -> logistic regression
                                                                              |
                          confidence >= threshold ?  yes -> answer from faqs.json
                                                     no  -> fallback reply + log the question
```

1. **Cleaning (`chatbot/preprocess.py`).** NLTK lowercases, removes punctuation, tokenizes, drops filler words and lemmatizes ("fees" becomes "fee"). Negations and question words ("not", "when", "how") are kept because they change the meaning.
2. **Intent classifier (`chatbot/classifier.py`).** Word and character TF-IDF features feed a logistic regression model trained on the examples in `faqs.json`.
3. **Confidence threshold.** Below the threshold (default 0.35) the bot uses the fallback reply.
4. **Bot logic (`chatbot/bot.py`).** Splits multi-question messages, avoids repeating answers and logs unanswered questions to a CSV file.
5. **Bot Doctor (`chatbot/doctor.py`).** Uses cross-validation to estimate accuracy, finds confusing and similar intents, and groups unanswered questions.

## Project structure

```
app.py                 Streamlit chat page
pages/Bot_Doctor.py    Streamlit Bot Doctor page
chatbot/
  preprocess.py        text cleaning (NLTK)
  classifier.py        intent classifier (scikit-learn)
  bot.py               multi-question handling, logging
  logger.py            unanswered-question and feedback logs (CSV)
  doctor.py            quality checks for the Bot Doctor
data/faqs.json         the bot's knowledge: intents, examples, answers
tests/test_bot.py      automated tests (pytest)
.streamlit/config.toml theme colours
requirements.txt       packages for deployment
```

## Run it on your computer

```bash
git clone https://github.com/shravya-181/gyanavedha-faq-chatbot.git
cd gyanavedha-faq-chatbot

python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt

streamlit run app.py
```

The NLTK word lists download automatically the first time the app starts.

## Run the tests

```bash
python -m pytest -v
```

19 tests check text cleaning, the classifier, multi-question splitting, the fallback, logging and the format of `faqs.json`.

## Use it for your own business

Edit `data/faqs.json`. Each intent has a name, example questions and an answer:

```json
{
  "business_name": "Your Business",
  "fallback_reply": "Sorry, I'm not sure about that yet.",
  "confidence_threshold": 0.35,
  "tagline": "Short line shown under the name",
  "suggestions": ["Fees", "Timings", "Contact"],
  "intents": [
    {
      "name": "timings",
      "examples": ["when are you open", "opening hours", "what time do you close"],
      "answer": "We are open Monday to Saturday, 9 AM to 6 PM."
    }
  ]
}
```

Tips: write at least 8 varied examples per intent, in different styles, and use the Bot Doctor to see where to add more.

## Results

On the sample FAQ file in this repository (13 intents, about 130 examples), the Bot Doctor reports around **74% accuracy on sentences the model has not seen** (5-fold cross-validation). At the default threshold, out-of-scope questions such as "what is the capital of France" are refused. Adding more varied examples to the weakest intents raises the score.

## Limitations

- It is an intent-matching bot, not a generative one. It can only answer what is in `faqs.json`.
- On Streamlit Community Cloud the log files (unanswered questions, feedback) reset when the app restarts.
- Accuracy depends on how many varied examples each intent has.
- Hinglish support is limited to character-level matching.

## Tech stack

Python, NLTK, scikit-learn, pandas, Streamlit, pytest.

## Author

Shravya M S
