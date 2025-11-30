AI Communication Scorer

An AI-powered tool that automatically evaluates a student's self-introduction speech based on a rubric with communication skills framework.

This project combines:

Rule-based checks (keywords, salutation, flow)

Statistical features (TTR, WPM, filler word rate)

Sentiment analysis (engagement)

Semantic similarity (SBERT sentence embeddings)

A clean Streamlit web interface

It produces a 0–100 score with detailed breakdowns and feedback. 


AI_Communication_Scorer
│
├── app.py                # Streamlit frontend UI
├── scoring.py            # Core scoring engine (Rubric + NLP + heuristics)
├── text_utils.py         # Preprocessing, tokenization, keyword detection
├── requirements.txt       # Python dependencies
└── Sample text for case study.txt   # Example transcript (optional) 


🚀 Features 

✔ 1. Transcript Input

Paste any self-introduction speech text and enter the duration (seconds).


✔ 2. Rule-Based Content Evaluation

Checks for:

Name

Age

School/Class

Family

Interests/Hobbies 

Good-to-have elements like plans, achievement, fun facts

Each contributes to Content & Structure score (0–40).


✔ 3. Speech Rate Scoring

Computes WPM (Words Per Minute) and maps to:

Ideal (111–140 WPM)

Slightly slow/fast

Too slow/too fast


✔ 4. Language & Grammar

Instead of using external grammar APIs (which fail due to Java/API limits), grammar is scored using a lightweight heuristic, checking:

Misuse of lowercase “ i ”

Double spaces

Strange tokens with numbers & letters

Basic structural errors

Vocabulary richness is computed using Type-Token Ratio (TTR). 


✔ 5. Clarity

Penalizes excessive filler words like:

um, uh, like, actually, basically, you know, kinda, i mean, well, hmm


✔ 6. Engagement

Uses VADER sentiment analysis to detect positivity / enthusiasm.


✔ 7. Semantic Similarity (NLP)

Each transcript is compared with 4 "ideal rubric descriptions":

Content & Structure

Language

Clarity

Engagement

Using sentence-transformers (all-MiniLM-L6-v2):

similarity = cos_sim(embedding(transcript), embedding(ideal_description))

These values (0–1) appear in the UI for interpretability.


✔ 8. Feedback Summary

Strengths (top 2 scoring areas) 

Areas for improvement (bottom 2 scoring areas)

Missing content elements 


📊 Scoring Formula (0–100)

1. Content & Structure – 40 Points

| Component                | Points |
| ------------------------ | ------ |
| Salutation               | 0–5    |
| Keyword coverage         | 0–30   |
| Flow / logical structure | 0–5    | 


2. Speech Rate – 10 Points

Based on WPM:

Ideal (111–140): 10

Slightly slow/fast: 6

Too slow/too fast: 2


3. Language & Grammar – 20 Points

Grammar heuristic → 0–10

Vocabulary richness (TTR) → 0–10


4. Clarity – 15 Points

Penalty for high filler word rate. 


5. Engagement – 15 Points

Sentiment positivity (VADER). 


Total = 40 + 10 + 20 + 15 + 15 = 100

🔮 Future Enhancements

Add live speech input (ASR → transcript → scoring)

Add multilingual scoring

Direct grammar API integration (paid or self-hosted LanguageTool)

Teacher dashboard for batch scoring

Real-time scoring during practice sessions

📝 Author

Developed as part of the Nirmaan Foundation AI Case Study.

