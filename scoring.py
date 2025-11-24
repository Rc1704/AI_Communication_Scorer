from typing import Dict, List, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

from text_utils import (
    preprocess_text,
    get_basic_stats,
    compute_ttr,
    count_filler_words,
    detect_salutation_level,
    detect_keywords,
    detect_structure_tags,
)

# -----------------------------
# External tools (sentiment + embeddings only)
# -----------------------------
analyzer = SentimentIntensityAnalyzer()
sem_model = SentenceTransformer("all-MiniLM-L6-v2")

# Short "ideal" descriptions for each high-level rubric dimension
CRITERION_DESCRIPTIONS = {
    "content": (
        "A well-structured self introduction with a clear salutation, name, age, "
        "class or school, family background, hobbies or interests, and a closing thank you."
    ),
    "language": (
        "Clear, grammatically correct English with appropriate sentence structure and "
        "a reasonably varied vocabulary for a school student."
    ),
    "clarity": (
        "Fluent and easy to understand speech with minimal filler words, concise sentences, "
        "and ideas expressed in a straightforward way."
    ),
    "engagement": (
        "A positive, enthusiastic, and friendly tone that feels genuine and engaging, "
        "making the listener interested in the speaker."
    ),
}

# Pre-compute embeddings for the criterion descriptions
CRITERION_EMBEDDINGS = {
    key: sem_model.encode(desc, convert_to_tensor=True)
    for key, desc in CRITERION_DESCRIPTIONS.items()
}


# -----------------------------
# Rule-based scoring functions
# -----------------------------
def score_salutation(text: str) -> int:
    """
    Salutation score: 0, 2, 4, or 5 based on greeting quality.
    Delegated to detect_salutation_level from text_utils.
    """
    return detect_salutation_level(text)


def score_keywords(text: str) -> Tuple[int, List[str], List[str]]:
    """
    Keyword coverage according to rubric.

    Must-haves (4 pts each, total 20):
      - Name
      - Age
      - School/Class
      - Family
      - Hobbies/Interests

    Good-to-haves (2 pts each, total 10):
      - About family (details)
      - Location/Origin
      - Ambition/Goal/Dream
      - Fun fact / Unique thing
      - Strengths/Achievements

    Returns:
      score_out_of_30,
      present_concepts (labels),
      missing_concepts (labels)
    """
    kw = detect_keywords(text)

    must_haves = [
        ("Name", "has_name"),
        ("Age", "has_age"),
        ("School/Class", "has_school_class"),
        ("Family", "has_family"),
        ("Hobbies/Interests", "has_hobbies"),
    ]
    good_to_have = [
        ("About family (details)", "has_about_family"),
        ("Location/Origin", "has_location"),
        ("Ambition/Goal/Dream", "has_ambition"),
        ("Fun fact / Unique thing", "has_fun_fact"),
        ("Strengths/Achievements", "has_strengths_or_achievements"),
    ]

    score = 0
    present = []
    missing = []

    # Must-haves: 4 points each
    for label, key in must_haves:
        if kw.get(key, False):
            score += 4
            present.append(label)
        else:
            missing.append(label)

    # Good-to-haves: 2 points each
    for label, key in good_to_have:
        if kw.get(key, False):
            score += 2
            present.append(label)
        else:
            missing.append(label)

    return score, present, missing


def score_flow(sentences: List[str], tags: List[str]) -> int:
    """
    Flow / Order (0 or 5).

    Simple, forgiving rule:
    - Has at least one SALUTATION
    - Has at least one BASIC sentence (name/age/class/place)
    - Has at least one CLOSING (thank you / that's all, etc.)

    If all three exist -> 5, else -> 0.
    """
    if not sentences or not tags:
        return 0

    has_sal = "SALUTATION" in tags
    has_basic = "BASIC" in tags
    has_closing = "CLOSING" in tags

    if has_sal and has_basic and has_closing:
        return 5
    return 0


def score_speech_rate(total_words: int, duration_sec: float) -> Tuple[int, float]:
    """
    Speech rate / WPM according to rubric bands:

      Too slow (< 80) or too fast (>161) -> 2
      Slow (81–110) or Fast (141–160)   -> 6
      Ideal (111–140)                   -> 10
    """
    if total_words == 0 or duration_sec <= 0:
        return 0, 0.0

    wpm = (total_words * 60.0) / duration_sec

    if wpm < 80 or wpm > 161:
        score = 2
    elif 81 <= wpm <= 110 or 141 <= wpm <= 160:
        score = 6
    elif 111 <= wpm <= 140:
        score = 10
    else:
        score = 2

    return score, wpm


def score_grammar(text: str, total_words: int) -> Tuple[int, float]:
    """
    Grammar scoring – local heuristic version (no external API).

    We approximate "error density" using a few rough indicators:
      - lowercase ' i ' used as pronoun
      - double spaces
      - strange tokens mixing letters and digits

    Then:
      errors_per_100 = (estimated_errors / total_words) * 100
      gram_frac = 1 - min(errors_per_100 / 20, 1)

    gram_frac mapped to bands:
        > 0.8          -> 10
        0.6–0.8        -> 8
        0.4–0.59       -> 6
        0.2–0.39       -> 4
        < 0.2          -> 2
    """
    if total_words == 0:
        return 0, 0.0

    lower_text = text.lower()

    # heuristic "errors"
    lower_i_errors = lower_text.count(" i ")  # lowercase i as standalone pronoun
    double_space_errors = text.count("  ")

    weird_tokens = [
        tok
        for tok in text.split()
        if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok)
    ]
    weird_punc_errors = len(weird_tokens)

    errors_est = lower_i_errors + double_space_errors + weird_punc_errors
    errors_per_100 = min((errors_est / max(total_words, 1)) * 100.0, 100.0)

    gram_frac = 1 - min(errors_per_100 / 20.0, 1.0)

    if gram_frac > 0.8:
        score = 10
    elif 0.6 <= gram_frac <= 0.8:
        score = 8
    elif 0.4 <= gram_frac < 0.6:
        score = 6
    elif 0.2 <= gram_frac < 0.4:
        score = 4
    else:
        score = 2

    return score, errors_per_100


def score_vocabulary(total_words: int, distinct_words: int) -> Tuple[int, float]:
    """
    Vocabulary richness (TTR) bands:

      0.9–1.0  -> 10
      0.7–0.89 -> 8
      0.5–0.69 -> 6
      0.3–0.49 -> 4
      0–0.29   -> 2
    """
    if total_words == 0:
        return 0, 0.0

    ttr = compute_ttr(total_words, distinct_words)

    if 0.9 <= ttr <= 1.0:
        score = 10
    elif 0.7 <= ttr < 0.9:
        score = 8
    elif 0.5 <= ttr < 0.7:
        score = 6
    elif 0.3 <= ttr < 0.5:
        score = 4
    else:
        score = 2

    return score, ttr


def score_clarity(text: str, total_words: int) -> Tuple[int, float, int]:
    """
    Clarity via filler word rate.

    Filler rate (percentage) bands:
      0–3   -> 15
      4–6   -> 12
      7–9   -> 9
      10–12 -> 6
      13+   -> 3
    """
    if total_words == 0:
        return 0, 0.0, 0

    fillers = [
        "um",
        "uh",
        "like",
        "you know",
        "so",
        "actually",
        "basically",
        "right",
        "i mean",
        "well",
        "kinda",
        "sort of",
        "okay",
        "hmm",
        "ah",
    ]

    filler_count = count_filler_words(text, fillers)
    filler_rate = (filler_count / total_words) * 100.0

    if 0 <= filler_rate <= 3:
        score = 15
    elif 4 <= filler_rate <= 6:
        score = 12
    elif 7 <= filler_rate <= 9:
        score = 9
    elif 10 <= filler_rate <= 12:
        score = 6
    else:
        score = 3

    return score, filler_rate, filler_count


def score_engagement(text: str) -> Tuple[int, float]:
    """
    Engagement via sentiment (VADER).

    Use 'pos' probability (0–1) and map:

      >= 0.7        -> 15
      0.5–0.69      -> 12
      0.3–0.49      -> 9
      0.1–0.29      -> 6
      < 0.1         -> 3
    """
    scores = analyzer.polarity_scores(text)
    pos = scores.get("pos", 0.0)

    if pos >= 0.7:
        score = 15
    elif 0.5 <= pos < 0.7:
        score = 12
    elif 0.3 <= pos < 0.5:
        score = 9
    elif 0.1 <= pos < 0.3:
        score = 6
    else:
        score = 3

    return score, pos


# -----------------------------
# Semantic similarity helpers
# -----------------------------
def compute_semantic_similarity(text: str, criterion_key: str) -> float:
    """
    Compute cosine similarity between the transcript and the
    ideal description for a given high-level criterion.

    Returns a value in [0, 1].
    """
    if criterion_key not in CRITERION_EMBEDDINGS:
        return 0.0

    text_emb = sem_model.encode(text, convert_to_tensor=True)
    crit_emb = CRITERION_EMBEDDINGS[criterion_key]
    sim = util.pytorch_cos_sim(text_emb, crit_emb).item()

    sim = max(0.0, min(1.0, sim))
    return sim


# -----------------------------
# Master scoring function
# -----------------------------
def score_transcript(text: str, duration_sec: float) -> Dict:
    """
    Main function: scores transcript according to the rubric.

    Components:
      - Content & Structure: salutation (5) + keywords (30) + flow (5) = 40
      - Speech Rate: 10
      - Language & Grammar: grammar (10) + vocab/TTR (10) = 20
      - Clarity: 15
      - Engagement: 15

    Total: 100 points.

    PLUS: Semantic similarity values for each high-level dimension
           (content, language, clarity, engagement) using sentence embeddings.
    """
    clean_text = preprocess_text(text)
    stats = get_basic_stats(clean_text)
    tokens = stats["tokens"]
    total_words = stats["total_words"]
    distinct_words = stats["distinct_words"]
    sentences = stats["sentences"]
    sentence_count = stats["sentence_count"]

    # Content & Structure
    salutation_score = score_salutation(clean_text)
    keyword_score, present_kw, missing_kw = score_keywords(clean_text)
    tags = detect_structure_tags(sentences)
    flow_score = score_flow(sentences, tags)

    # Speech rate
    speech_score, wpm = score_speech_rate(total_words, duration_sec)

    # Grammar & Vocabulary (heuristic grammar, local TTR)
    grammar_score, errors_per_100 = score_grammar(text, total_words)
    vocab_score, ttr = score_vocabulary(total_words, distinct_words)

    # Clarity & Engagement
    clarity_score, filler_rate, filler_count = score_clarity(clean_text, total_words)
    engagement_score, pos_prob = score_engagement(text)

    total_score = (
        salutation_score
        + keyword_score
        + flow_score
        + speech_score
        + grammar_score
        + vocab_score
        + clarity_score
        + engagement_score
    )

    # Semantic similarities (0–1) for each major dimension
    content_sem = compute_semantic_similarity(clean_text, "content")
    language_sem = compute_semantic_similarity(clean_text, "language")
    clarity_sem = compute_semantic_similarity(clean_text, "clarity")
    engagement_sem = compute_semantic_similarity(clean_text, "engagement")

    result = {
        "total_score": total_score,
        "stats": {
            "total_words": total_words,
            "distinct_words": distinct_words,
            "sentence_count": sentence_count,
            "tokens": tokens,
        },
        "wpm": wpm,
        "salutation_score": salutation_score,
        "keyword_score": keyword_score,
        "present_keywords": present_kw,
        "missing_keywords": missing_kw,
        "flow_score": flow_score,
        "speech_score": speech_score,
        "grammar_score": grammar_score,
        "errors_per_100": errors_per_100,
        "vocab_score": vocab_score,
        "ttr": ttr,
        "clarity_score": clarity_score,
        "filler_rate": filler_rate,
        "filler_count": filler_count,
        "engagement_score": engagement_score,
        "pos_prob": pos_prob,
        "tags": tags,
        "content_semantic": content_sem,
        "language_semantic": language_sem,
        "clarity_semantic": clarity_sem,
        "engagement_semantic": engagement_sem,
    }

    return result


if __name__ == "__main__":
    sample_text = (
        "Hello everyone, my name is Arjun. I am 14 years old and I study in "
        "Class 9 at Sunrise Public School. I live in Bangalore with my parents "
        "and my younger sister. In my free time, I enjoy playing football and "
        "reading stories. Thank you for listening."
    )
    res = score_transcript(sample_text, duration_sec=60)
    print("Total score:", res["total_score"])
    print(
        "Semantic content / language / clarity / engagement:",
        res["content_semantic"],
        res["language_semantic"],
        res["clarity_semantic"],
        res["engagement_semantic"],
    )
