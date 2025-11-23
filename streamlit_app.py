import os

import streamlit as st

from ml_pipeline import load_model, predict_comment, label_map


def main():
    st.set_page_config(page_title="Toxic Comment Detection", layout="wide")
    st.title("🧪 Toxic Comment Detection (EN + RU)")

    st.markdown(
        """
This demo uses a multilingual (English + Russian) ML model based on TF-IDF and
multi-label Logistic Regression. It can detect several toxicity types:
general toxicity, hate speech, sexism, homophobia, etc.
"""
    )

    load_model()

    text = st.text_area(
        "Enter a comment (English or Russian):",
        height=150,
        placeholder="Type any social media comment here...",
    )

    col1, col2 = st.columns([1, 2])

    if col1.button("Analyze"):
        if not text.strip():
            st.warning("Please enter a non-empty comment.")
            return

        label, prob, toxic_types = predict_comment(text, visualize=True)

        col1.subheader("Prediction")
        col1.write(f"**Result:** {label}")
        col1.write(f"**Overall confidence:** {prob:.4f}")

        if toxic_types:
            col1.write("**Detected toxicity types:**")
            for code, p in toxic_types:
                human_name = label_map.get(code, code)
                col1.write(f"- {human_name} (probability {p:.2f})")
        else:
            if label == "Toxic":
                col1.write(
                    "Model marked the comment as toxic, "
                    "but did not activate specific subtypes."
                )
            else:
                col1.write("No specific toxicity types detected.")

        # Show probability bar chart (saved by ml_pipeline)
        if os.path.exists("last_prediction_probs.png"):
            col2.subheader("Per-label probabilities")
            col2.image("last_prediction_probs.png", use_column_width=True)

    st.markdown("---")
    st.subheader("Model diagnostics")

    diag_cols = st.columns(3)

    if os.path.exists("label_f1_scores.png"):
        diag_cols[0].image("label_f1_scores.png", caption="F1-score by label", use_column_width=True)

    if os.path.exists("confusion_is_toxic.png"):
        diag_cols[1].image("confusion_is_toxic.png", caption="Confusion matrix: IsToxic", use_column_width=True)

    if os.path.exists("wordcloud_toxic.png"):
        diag_cols[2].image("wordcloud_toxic.png", caption="Toxic comments wordcloud", use_column_width=True)


if __name__ == "__main__":
    main()
