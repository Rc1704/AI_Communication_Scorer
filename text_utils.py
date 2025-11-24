import re
from typing import List, Dict


def preprocess_text(text: str) -> str:
    """
    Lowercase and strip extra whitespace.
    """
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def get_basic_stats(text: str) -> Dict:
    """
    Compute basic stats: tokens, counts, sentences.
    """
    clean = text.strip()
    if not clean:
        return {
            "tokens": [],
            "total_words": 0,
            "distinct_words": 0,
            "sentences": [],
            "sentence_count": 0,
        }

    # simple tokenization
    tokens = clean.split()
    total_words = len(tokens)
    distinct_words = len(set(tokens))

    # sentence split on punctuation
    raw_sentences = re.split(r"[.!?]+", clean)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    sentence_count = len(sentences)

    return {
        "tokens": tokens,
        "total_words": total_words,
        "distinct_words": distinct_words,
        "sentences": sentences,
        "sentence_count": sentence_count,
    }


def compute_ttr(total_words: int, distinct_words: int) -> float:
    if total_words == 0:
        return 0.0
    return distinct_words / total_words


def count_filler_words(text: str, fillers: List[str]) -> int:
    """
    Count filler words (single and multi-word).
    """
    if not text:
        return 0

    text_lower = text.lower()
    # For counting single-word fillers, use tokens
    tokens = text_lower.split()
    token_counts = {}
    for t in tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    count = 0
    for f in fillers:
        f = f.lower().strip()
        if " " in f:
            # multi-word phrase: approximate with substring count
            count += text_lower.count(f)
        else:
            count += token_counts.get(f, 0)

    return count


def detect_salutation_level(text: str) -> int:
    """
    Returns 0, 2, 4, or 5 based on first sentence.
    """
    if not text:
        return 0

    stats = get_basic_stats(text)
    sentences = stats["sentences"]
    if not sentences:
        return 0

    first = sentences[0].lower()

    # Most enthusiastic
    enthusiastic_phrases = [
        "excited to introduce myself",
        "thrilled to introduce myself",
        "thrilled to be here",
        "excited to be here",
        "i am excited to introduce",
        "i'm excited to introduce",
    ]
    for p in enthusiastic_phrases:
        if p in first:
            return 5

    # Good formal greetings
    formal_greetings = [
        "good morning",
        "good afternoon",
        "good evening",
        "hello everyone",
        "hello everybody",
    ]
    for p in formal_greetings:
        if p in first:
            return 4

    # Simple greetings
    simple_greetings = [
        "hello",
        "hi",
        "hey",
    ]
    for p in simple_greetings:
        # avoid double-counting "hello everyone" which we handled above
        if p in first:
            return 2

    return 0


def detect_keywords(text: str) -> Dict[str, bool]:
    """
    Detect presence of rubric 'concepts' using simple patterns.
    """
    t = text.lower()

    # Must-have
    has_name = any(p in t for p in ["my name is", "myself", "i am "])
    has_age = "years old" in t
    has_school_class = any(p in t for p in ["class ", "standard", "grade ", "school"])
    has_family = any(
        p in t for p in ["family", "mother", "father", "parents", "brother", "sister"]
    )
    has_hobbies = any(
        p in t
        for p in [
            "my hobby is",
            "my hobbies are",
            "i like to",
            "i love to",
            "i enjoy",
            "in my free time",
        ]
    )

    # Good-to-have
    # about family - more than just one word "family"
    has_about_family = any(
        p in t
        for p in [
            "my family is",
            "we are a family of",
            "there are",
            "members in my family",
        ]
    )
    has_location = any(p in t for p in ["i am from", "i'm from", "i live in", "my hometown"])
    has_ambition = any(
        p in t
        for p in [
            "i want to become",
            "i want to be",
            "my dream is",
            "my goal is",
            "my ambition is",
        ]
    )
    has_fun_fact = any(
        p in t
        for p in [
            "fun fact",
            "something unique about me",
            "one thing about me",
            "an interesting thing about me",
        ]
    )
    has_strengths_or_achievements = any(
        p in t
        for p in [
            "i am good at",
            "i'm good at",
            "my strength is",
            "my strengths are",
            "i have won",
            "i won",
            "i achieved",
            "i have achieved",
        ]
    )

    return {
        "has_name": has_name,
        "has_age": has_age,
        "has_school_class": has_school_class,
        "has_family": has_family,
        "has_hobbies": has_hobbies,
        "has_about_family": has_about_family,
        "has_location": has_location,
        "has_ambition": has_ambition,
        "has_fun_fact": has_fun_fact,
        "has_strengths_or_achievements": has_strengths_or_achievements,
    }


def _sentence_has_basic(s: str) -> bool:
    s = s.lower()
    if any(p in s for p in ["my name is", "myself", "i am "]):
        return True
    if "years old" in s:
        return True
    if any(p in s for p in ["class ", "standard", "grade ", "school"]):
        return True
    if any(p in s for p in ["i am from", "i live in", "i'm from", "my hometown"]):
        return True
    return False


def _sentence_has_additional(s: str) -> bool:
    s = s.lower()
    if any(p in s for p in ["family", "mother", "father", "parents", "brother", "sister"]):
        return True
    if any(
        p in s
        for p in [
            "my hobby is",
            "my hobbies are",
            "i like to",
            "i love to",
            "i enjoy",
            "in my free time",
        ]
    ):
        return True
    if any(
        p in s
        for p in [
            "i want to become",
            "i want to be",
            "my dream is",
            "my goal is",
            "my ambition is",
        ]
    ):
        return True
    if any(
        p in s
        for p in [
            "fun fact",
            "something unique about me",
            "one thing about me",
            "an interesting thing about me",
        ]
    ):
        return True
    if any(
        p in s
        for p in [
            "i am good at",
            "i'm good at",
            "my strength is",
            "my strengths are",
            "i have won",
            "i won",
            "i achieved",
            "i have achieved",
        ]
    ):
        return True
    return False


def _sentence_has_salutation(s: str) -> bool:
    s = s.lower()
    if any(p in s for p in ["good morning", "good afternoon", "good evening"]):
        return True
    if any(p in s for p in ["hello", "hi", "hey"]):
        return True
    if any(
        p in s
        for p in [
            "excited to introduce myself",
            "thrilled to introduce myself",
            "excited to be here",
            "thrilled to be here",
        ]
    ):
        return True
    return False


def _sentence_has_closing(s: str) -> bool:
    s = s.lower()
    if "thank you" in s or "thanks for listening" in s or "that's all" in s:
        return True
    return False


def detect_structure_tags(sentences: List[str]) -> List[str]:
    """
    Tag each sentence as SALUTATION / BASIC / ADDITIONAL / CLOSING / OTHER.
    """
    tags = []
    for s in sentences:
        if _sentence_has_salutation(s):
            tags.append("SALUTATION")
        elif _sentence_has_closing(s):
            tags.append("CLOSING")
        elif _sentence_has_basic(s):
            tags.append("BASIC")
        elif _sentence_has_additional(s):
            tags.append("ADDITIONAL")
        else:
            tags.append("OTHER")
    return tags
