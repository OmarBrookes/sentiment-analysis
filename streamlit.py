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
    "I absolutely love this! Everything works perfectly.",        # positive
    "This is the worst product I’ve ever bought.",                # negative
    "It’s fine. Not good, not bad. Just fine.",                   # neutral
    "The product is great, but the customer service was awful.",  # mixed
    "Yeah, this is exactly what I needed... not.",                # sarcastic & negative
    "Fantastic quality and super fast shipping!",                 # positive
    "Do not waste your money on this junk.",                      # negative
    "It does what it says, but I wouldn’t buy it again.",         # mixed
    "Oh great, another feature that doesn’t actually work.",      # sarcastic
    "Pretty average overall, nothing stood out.",                 # neutral
    "If disappointment had a face, it would be this product.",    # sarcastic, negative & ironic
    "Good build, but the software is a nightmare.",               # mixed
    "I’m happy with my purchase.",                                # positive
    "It broke after two days. Useless.",                          # negative
    "Nothing special, it’s just okay.",                           # neutral
    "Nice design, awful performance.",                            # mixed
    "Oh sure, because *that* feature really helped... not.",      # sarcastic
    "This thing really nailed the 'barely works' vibe.",          # sarcastic & ironic
]

# -=-=-=- Streamlit UI Setup -=-=-=-
st.title("💬 Sentiment Analysis")
st.write("Enter text or upload a file to get sentiment predictions.")

# Session state tracking
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "analyse_clicked" not in st.session_state:
    st.session_state.analyse_clicked = False
if "review_count" not in st.session_state:
    st.session_state.review_count = 0

# Input field
user_input = st.text_area("Enter text here:", key="input_text")

# File upload
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"], key="file_uploader")

# Buttons layout
st.markdown("###")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Analyse Sentiment"):
        st.session_state.analyse_clicked = True

with col2:
    if st.button("🗑️ Clear"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.review_count = 0
        st.rerun()

with col3:
    if "shuffled_samples" not in st.session_state or not st.session_state.shuffled_samples:
        st.session_state.shuffled_samples = random.sample(SAMPLE_REVIEWS, len(SAMPLE_REVIEWS))

    if st.button("🎲 Try Sample Review"):
        for key in list(st.session_state.keys()):
            if key not in ["shuffled_samples"]:
                del st.session_state[key]

        new_sample = st.session_state.shuffled_samples.pop()
        st.session_state.input_text = new_sample
        st.session_state.analyse_clicked = True
        st.rerun()

# Prediction function
def analyse_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    predicted_labels = [label for label, prob in zip(LABELS, probs) if prob >= custom_thresholds[label]]

    if "positive" in predicted_labels and "negative" in predicted_labels and "mixed" not in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]
        predicted_labels.append("mixed")

    if "mixed" in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]

    return predicted_labels if predicted_labels else ["neutral"]

# Analyse from text input (NO numbering here)
if st.session_state.analyse_clicked:
    if st.session_state.input_text:
        with st.spinner("🔍 Analysing..."):
            user_input_single_line = " ".join(st.session_state.input_text.splitlines())
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
    st.session_state.analyse_clicked = False

# Analyse from uploaded file (WITH numbering)
if uploaded_file:
    sentences = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
    st.subheader(f"Processing {len(sentences)} reviews from file...")
    for idx, sentence in enumerate(sentences, start=1):
        st.session_state.review_count += 1
        sentiment = analyse_sentiment(sentence)
        sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
        sentiment_colour = COLOURS[sentiment[0]]

        st.markdown(
            f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review #{st.session_state.review_count}: "{sentence}"</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:10px;">Sentiment: {sentiment_with_emojis}</div>',
            unsafe_allow_html=True
        )
        for label in sentiment:
            st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)
        st.markdown("---")
