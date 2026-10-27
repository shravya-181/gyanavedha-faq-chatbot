# Gyanavedha Institute: FAQ Chatbot with a Built-in "Bot Doctor"

A customer-support chatbot for a UG and PG college. It answers common questions about admissions, fees, courses, exams, hostel, scholarships, placements and more, remembers which course a student is asking about, and tells the college office **how to make itself better**.

**Live demo:** https://gyanavedha-institute.streamlit.app

> Gyanavedha Institute is a **fictional demo college**. All fees, dates, phone numbers and e-mail addresses in this project are sample data.

![Chatbot screenshot](docs/screenshot.png)

---

## What makes it different

Most student chatbot projects stop at "type a question, get an answer". This one goes further:

| Feature | What it does |
|---|---|
| **Plug-and-play FAQ file** | All knowledge lives in one file, `data/faqs.json`. Anyone can create a bot for their own business by editing that file. No Python needed. |
| **Course-aware answers** | Ask "BBA fees" and get BBA's own fee, duration and eligibility. The bot remembers the course ("what about MBA?" repeats the topic for MBA) and answers for UG or PG when asked. A **Forget** button or the word `reset` clears the memory. |
| **Bot Doctor** | A staff-only page that tests the bot on sentences it has never seen and reports its accuracy, which intents get mixed up, which are weak, and what visitors asked that the bot could not answer. It is protected by a staff password. |
| **Honest accuracy test** | A separate set of fresh questions (`data/holdout.json`) that the bot never trains on. An automated test fails if any of them leak into the training data. |
| **Multi-question messages** | "fees and hostel? is there an annual fest?" is split into separate questions. Each one is answered, and unknown ones are logged. |
| **Safe fallback** | When the bot is not confident, it says so instead of guessing. A confidence threshold decides. |
| **Unanswered-question log** | Every question the bot could not answer is saved, so the owner knows what to add next. |
| **Thumbs up / thumbs down** | Visitors rate each answer once. Unhelpful answers appear in the Bot Doctor. |
| **Typo and Hinglish tolerant** | Character-level features and Hinglish training examples help with spelling mistakes ("accomodation", "placments") and mixed-language questions ("fees kitna hai"). |

---

## How it works

```
Customer message
      |
      v
 split into questions        (bot.py)
      |
      v
 find course / UG-PG         (bot.py: "BBA", "B.Com", "after 12th" ...
      |                       remembered for the rest of the chat)
      v
 clean the text              (preprocess.py: lowercase, remove punctuation,
      |                       tokenize, remove filler words, lemmatize)
      v
 TF-IDF features             (word 1-2 grams + character 2-5 grams)
      |
      v
 Logistic regression         (classifier.py: probability for every intent)
      |
      v
 confidence >= threshold?
   yes -> reply with the course-specific, level-specific or general answer
   no  -> safe fallback reply + log the question (logger.py)
```

Some design choices worth knowing:

- **Negations and question words are kept** during cleaning. "can't pay" must not become "pay", and "when" matters in "when are the exams".
- **Character n-grams** were added after testing showed that word features alone gave timid, low-confidence predictions on paraphrases and typos.
- **The threshold (0.45)** was chosen by comparing it with other values on fresh questions and off-topic questions. At 0.45 the bot refused about 77% of off-topic questions while giving up almost none of the real ones.
- **Accuracy is measured two ways.** The Bot Doctor cross-validates (trains on part of the examples, tests on the rest). The fresh-question test uses sentences that were never in the training file.

---

## Project structure

```
gyanavedha-faq-chatbot/
├── app.py                  # Streamlit chat page
├── pages/
│   └── Bot_Doctor.py       # staff-only Bot Doctor page
├── chatbot/
│   ├── preprocess.py       # text cleaning with NLTK
│   ├── classifier.py       # TF-IDF + logistic regression, confidence threshold
│   ├── bot.py              # multi-question handling, course memory, replies
│   ├── logger.py           # unanswered-question and feedback logs (CSV)
│   ├── doctor.py           # diagnosis engine for the Bot Doctor
│   └── evaluate.py         # held-out evaluation, fresh-question test, dataset checks
├── data/
│   ├── faqs.json           # the bot's knowledge: programs, intents, examples, answers
│   ├── holdout.json        # fresh test questions the bot never trains on
│   └── oos.json            # off-topic questions the bot should refuse
├── tests/                  # 33 automated tests
├── .github/workflows/      # runs the tests on every push
├── .streamlit/config.toml  # theme colours
├── requirements.txt
└── requirements-dev.txt
```

---

## Run it on your computer

You need Python 3.10 or newer.

```bash
# 1. Get the code
git clone https://github.com/shravya-181/gyanavedha-faq-chatbot.git
cd gyanavedha-faq-chatbot

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate

# 3. Install the packages
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The first run downloads small NLTK word lists automatically. The app opens at `http://localhost:8501`. The **Bot Doctor** page is in the sidebar (the arrow at the top left).

### Run the tests and the evaluation

```bash
pip install -r requirements-dev.txt
python -m pytest -v
python -m chatbot.evaluate
```

### Protect the Bot Doctor with a staff password

The password is never stored in the code. On Streamlit Community Cloud, open **Settings → Secrets** and add:

```toml
STAFF_PASSWORD = "your-password-here"
```

For local testing, put the same line in `.streamlit/secrets.toml` (this file is ignored by git). If no password is set, the page stays open. Staff are logged out when they go back to the chat page.

---

## Use it for your own business

1. Open `data/faqs.json`.
2. Change `business_name`, `fallback_reply` and the optional `tagline`, `suggestions` and `contact`.
3. Replace the `intents`. Each intent has:
   - `name`: a short label, for example `fee_structure`
   - `examples`: 15 or more different ways a customer might ask (variety matters more than quantity)
   - `answer`: the reply to give. Markdown works (bold, lists, tables).
4. Optional: add `programs` and a `program_answer` or `level_answer` to an intent to get course-aware answers.
5. Restart the app and open the **Bot Doctor**. It will tell you which intents need more examples.

---

## Tech stack

Python, Streamlit, scikit-learn (TF-IDF, logistic regression, cross-validation), NLTK (tokenizing, stop words, lemmatizing), pandas, pytest, ruff, GitHub Actions.

---

## Results

The bot was tested in three ways, on a 19-intent FAQ file with about 460 training examples:

| Test | Result |
|---|---|
| **Fresh questions** (122 questions never used in training), best guess is the right intent | **92%** |
| Of the answers the bot actually gave on those questions, how many were right | 93% |
| Off-topic questions refused (30 questions, threshold 0.45) | 77% |
| Cross-validated split (70/30 held-out) | 84% |

How it improved: the first version scored about **67%** on a held-out split. The Bot Doctor pointed to the weak intents (admission, courses, contact) and to intents that were mistaken for each other. Adding varied examples, including typos and Hinglish, raised the score step by step.

**Honest notes**
- These numbers come from small test sets, so expect a point or two of change whenever the data changes.
- The fresh questions were written by the project author, so real visitors will word things in ways the test does not cover.
- An earlier round scored 100% only because the training examples had been tuned after looking at that test set. That test set was moved into training and replaced with new questions. The numbers above are from the new, untouched set.

---

## Limitations

- **Free hosting forgets the logs.** On Streamlit Community Cloud, the unanswered-question and feedback files are cleared when the app restarts. For permanent logs, a database would be the next step.
- **It only knows what the FAQ file teaches it.** It does not search the web or generate new answers, so it cannot answer questions outside its intents. That is deliberate, because it prevents confident wrong answers.
- **Some off-topic questions can still slip through** (about 1 in 4 in my test) if they happen to resemble an intent.
- **Hinglish support is basic.** It understands common Hinglish questions, but answers are in English.

---

## Roadmap

- A human-handoff form that saves the visitor's name and phone number for the office
- Full Hinglish support, including replies
- An admin page to edit the FAQ file inside the app

---

## Author

**Shravya M S**
GitHub: [@shravya-181](https://github.com/shravya-181)

Built during the 2-Month Artificial Intelligence Internship at DEC Technologies.
