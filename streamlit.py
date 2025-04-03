import streamlit as st
import torch
import transformers
import random
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# -=-=-=- Setup: Load model, tokenizer and set device -=-=-=-
MODEL_PATH = "OmarBrookes/sentiment-analysis"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label names, emojis, and colours
LABELS = ['neutral', 'positive', 'mixed', 'sarcastic', 'negative', 'ironic']

EMOJIS = {
    'neutral': '😐',
    'positive': '😊',
    'mixed': '🤔',
    'sarcastic': '🙃',
    'negative': '😞',
    'ironic': '😏'
}

COLOURS = {
    'neutral': '#D3D3D3',
    'positive': '#90EE90',
    'mixed': '#FFD700',
    'sarcastic': '#87CEEB',
    'negative': '#FFB6C1',
    'ironic': '#D8BFD8'
}

REVIEW_MESSAGES = {
    "positive": "✔️ This review is positive and expresses satisfaction.",
    "negative": "❌ This review is negative and shows dissatisfaction.",
    "neutral": "➖ This review is neutral and doesn’t show strong emotions.",
    "mixed": "🔀 This review contains both positive and negative feelings.",
    "sarcastic": "🙃 This review sounds sarcastic — it might mean the opposite of what it says.",
    "ironic": "😏 This review has an ironic tone — likely saying one thing but implying another."
}

# Custom thresholds
custom_thresholds = {
    'neutral': 0.39,
    'positive': 0.509,
    'mixed': 0.22,
    'sarcastic': 0.29,
    'negative': 0.428,
    'ironic': 0.16
}

# Sample reviews to randomly choose from
SAMPLE_REVIEWS = [
    "I absolutely love this! Everything works perfectly.",
    "This is the worst product I’ve ever bought.",
    "It’s fine. Not good, not bad. Just fine.",
    "The product is great, but the customer service was awful.",
    "Yeah, this is exactly what I needed... not.",
    "Fantastic quality and super fast shipping!",
    "Do not waste your money on this junk.",
    "It does what it says, but I wouldn’t buy it again.",
    "Oh great, another feature that doesn’t actually work.",
    "Pretty average overall, nothing stood out.",
    "If disappointment had a face, it would be this product.",
    "Honestly, I expected worse. Pleasantly surprised.",
    "Good build, but the software is a nightmare.",
    "Wow, just wow. Not in a good way though."
]

# -=-=-=- Streamlit UI Setup -=-=-=-
st.title("💬 Sentiment Analysis")
st.write("Enter text or upload a file to get sentiment predictions.")

# Session state tracking
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

review_count = 0
analyse_clicked = False

# Input field
user_input = st.text_area("Enter text here:", value=st.session_state.user_input, key="input_text")

# File upload
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"], key="file_uploader")

# Buttons layout
st.markdown("###")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Analyse Sentiment"):
        analyse_clicked = True

with col2:
    if st.button("🗑️ Clear"):
        st.session_state.user_input = ""
        if "input_text" in st.session_state:
            del st.session_state["input_text"]
        if "file_uploader" in st.session_state:
            del st.session_state["file_uploader"]
        st.experimental_rerun()

with col3:
    if st.button("🎲 Try Sample Review"):
        random_sample = random.choice(SAMPLE_REVIEWS)
        st.session_state.user_input = random_sample
        analyse_clicked = True

# Prediction function
def analyse_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    predicted_labels = [label for label, prob in zip(LABELS, probs) if prob >= custom_thresholds[label]]

    # Convert positive + negative to mixed if needed
    if "positive" in predicted_labels and "negative" in predicted_labels and "mixed" not in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]
        predicted_labels.append("mixed")

    # If mixed is present, remove positive and negative
    if "mixed" in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]

    return predicted_labels if predicted_labels else ["neutral"]

# Analyse from text input
if analyse_clicked:
    if user_input:
        with st.spinner("🔍 Analysing..."):
            review_count += 1
            st.session_state.user_input = user_input
            user_input_single_line = " ".join(user_input.splitlines())
            st.subheader(f"Review #{review_count}")
            st.markdown(
                f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review Entered: "{user_input_single_line}"</div>',
                unsafe_allow_html=True
            )

            sentiment = analyse_sentiment(user_input_single_line)
            sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
            sentiment_colour = COLOURS[sentiment[0]]

            st.markdown(
                f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:15px;">Sentiment: {sentiment_with_emojis}</div>',
                unsafe_allow_html=True
            )

            for label in sentiment:
                st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)

            st.markdown("---")

# Analyse from uploaded file
if uploaded_file:
    sentences = uploaded_file.read().decode("utf-8").splitlines()
    st.subheader(f"Processing {len(sentences)} reviews from file...")
    for idx, sentence in enumerate(sentences, start=1):
        if sentence.strip():
            review_count += 1
            sentiment = analyse_sentiment(sentence)
            sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
            sentiment_colour = COLOURS[sentiment[0]]

            st.markdown(
                f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review #{review_count}: "{sentence}"</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:10px;">Sentiment: {sentiment_with_emojis}</div>',
                unsafe_allow_html=True
            )
            for label in sentiment:
                st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)
            st.markdown("---")
