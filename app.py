import streamlit as st
import pandas as pd

from scoring import score_transcript


def label_overall(score: int) -> str:
    if score >= 90:
        return "Outstanding"
    elif score >= 75:
        return "Very good"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Needs improvement"
    else:
        return "Weak"


def main():
    st.title("AI-based Self-Introduction Scoring Tool")
    st.write(
        "Paste a student's self-introduction and get a detailed score based on a communication skills rubric.\n\n"
        "This tool combines rule-based checks (keywords, structure), statistical metrics (TTR, filler rate, WPM), "
        "and NLP-based semantic similarity using sentence embeddings."
    )

    st.sidebar.header("Input options")
    duration = st.sidebar.number_input(
        "Duration of speech (in seconds)", min_value=1, value=60
    )

    use_sample = st.sidebar.checkbox("Use sample transcript text")

    text = st.text_area("Paste transcript here", height=250)

    if use_sample:
        sample_text = (
            "Hello everyone, my name is Muskan. I am 13 years old and I study in "
            "class 8B at Christ Public School. We are a family of three and they are "
            "very kind and soft spoken. In my free time, I love playing cricket and "
            "sometimes talk to myself in the mirror. Thank you for listening."
        )
        if not text.strip():
            text = sample_text
            st.info("Sample transcript loaded. You can edit it if you like.")
            st.session_state["sample_loaded"] = True

    if st.button("Score my introduction"):
        if not text.strip():
            st.warning("Please paste a transcript first.")
        else:
            with st.spinner("Scoring transcript..."):
                res = score_transcript(text, duration)

            total = res["total_score"]
            label = label_overall(total)

            st.subheader("Overall Score")
            st.metric(label="Score (out of 100)", value=total)
            st.caption(f"Performance band: {label}")

            stats = res["stats"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", stats["total_words"])
            col2.metric("Sentences", stats["sentence_count"])
            col3.metric("WPM", round(res["wpm"], 1))

            # Read semantic similarities
            content_sem = res["content_semantic"]
            language_sem = res["language_semantic"]
            clarity_sem = res["clarity_semantic"]
            engagement_sem = res["engagement_semantic"]

            st.subheader("Detailed Breakdown")

            rows = [
                [
                    "Content & Structure",
                    "Salutation",
                    res["salutation_score"],
                    5,
                    round(content_sem, 3),
                ],
                [
                    "Content & Structure",
                    "Keyword coverage",
                    res["keyword_score"],
                    30,
                    round(content_sem, 3),
                ],
                [
                    "Content & Structure",
                    "Flow / Order",
                    res["flow_score"],
                    5,
                    round(content_sem, 3),
                ],
                [
                    "Speech Rate",
                    "Speech rate score",
                    res["speech_score"],
                    10,
                    "",  # semantic similarity not really meaningful here
                ],
                [
                    "Language & Grammar",
                    "Grammar",
                    res["grammar_score"],
                    10,
                    round(language_sem, 3),
                ],
                [
                    "Language & Grammar",
                    "Vocabulary richness (TTR)",
                    res["vocab_score"],
                    10,
                    round(language_sem, 3),
                ],
                [
                    "Clarity",
                    "Filler words",
                    res["clarity_score"],
                    15,
                    round(clarity_sem, 3),
                ],
                [
                    "Engagement",
                    "Sentiment positivity",
                    res["engagement_score"],
                    15,
                    round(engagement_sem, 3),
                ],
            ]

            df = pd.DataFrame(
                rows,
                columns=["Category", "Metric", "Score", "Max Score", "Semantic sim (0–1)"],
            )
            st.table(df)

            # Semantic summary
            st.subheader("Semantic Similarity Summary")
            st.write(
                f"- **Content & structure match:** {content_sem:.3f} (0–1, higher means closer to ideal introduction)"
            )
            st.write(
                f"- **Language quality match:** {language_sem:.3f} "
                "(how close the language feels to clear, grammatical English)"
            )
            st.write(
                f"- **Clarity match:** {clarity_sem:.3f} "
                "(alignment with fluent, low-filler, easy-to-follow speech)"
            )
            st.write(
                f"- **Engagement match:** {engagement_sem:.3f} "
                "(alignment with a positive, enthusiastic tone)"
            )

            # Keyword presence
            st.subheader("Content Coverage (Keywords)")
            present = res["present_keywords"]
            missing = res["missing_keywords"]

            col_p, col_m = st.columns(2)
            col_p.markdown("**Present:**")
            if present:
                for p in present:
                    col_p.write(f" {p}")
            else:
                col_p.write("None detected.")

            col_m.markdown("**Missing or weak:**")
            if missing:
                for m in missing:
                    col_m.write(f" {m}")
            else:
                col_m.write("All key areas covered!")

            # Feedback
            st.subheader("Feedback Summary")

            strengths = []
            improvements = []

            metric_perc = {
                "Content & Structure: Salutation": res["salutation_score"] / 5 if 5 else 0,
                "Content & Structure: Keywords": res["keyword_score"] / 30 if 30 else 0,
                "Content & Structure: Flow": res["flow_score"] / 5 if 5 else 0,
                "Speech rate": res["speech_score"] / 10 if 10 else 0,
                "Grammar": res["grammar_score"] / 10 if 10 else 0,
                "Vocabulary": res["vocab_score"] / 10 if 10 else 0,
                "Clarity": res["clarity_score"] / 15 if 15 else 0,
                "Engagement": res["engagement_score"] / 15 if 15 else 0,
            }

            sorted_metrics = sorted(
                metric_perc.items(), key=lambda x: x[1], reverse=True
            )
            top2 = sorted_metrics[:2]
            bottom2 = sorted_metrics[-2:]

            for name, val in top2:
                strengths.append(f"{name} is strong ({round(val*100)}%).")

            for name, val in bottom2:
                improvements.append(
                    f"{name} could be improved ({round(val*100)}%)."
                )

            st.markdown("**Strengths:**")
            if strengths:
                for s in strengths:
                    st.write(f"- {s}")
            else:
                st.write("- Not enough information to determine strengths.")

            st.markdown("**Areas for improvement:**")
            if improvements:
                for imp in improvements:
                    st.write(f"- {imp}")
            else:
                st.write("- Not enough information to determine improvements.")


if __name__ == "__main__":
    main()
