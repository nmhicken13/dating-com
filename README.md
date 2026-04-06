# Dating tracker (Streamlit)

A small local app to track people you are dating or talking to and log outings. Data is stored in SQLite on your machine (`dating_tracker.db` in this folder).

## Requirements

- Python 3.10+ recommended

## Setup

Create a virtual environment and install dependencies:

```bash
cd "/Users/natehicken/dating tracker"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Run the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

## What it does

- **Dashboard**: counts (people, active, total dates, last 30 days), charts by status and by month, and a recent-dates table.
- **People**: add someone (name, how you met, status, notes) and browse the list.
- **Log a date**: record an outing linked to a person, with optional activity, rating, and notes.

The database file is created automatically the first time you use the app. It is listed in `.gitignore` so your data is not committed to git by default.
