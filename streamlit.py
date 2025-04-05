import streamlit as st
import torch
import transformers
import random
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# -=-=-=- Setup: Load model, tokenizer and set device -=-=-=-
# Load the pre-trained RoBERTa model and tokenizer from Hugging Face Hub
MODEL_PATH = "OmarBrookes/sentiment-analysis"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# Automatically use GPU if available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -=-=-=- Configuration: Labels, Emojis, Colours, Messages -=-=-=-
# Define the possible sentiment categories the model can detect
LABELS = ['neutral', 'positive', 'mixed', 'sarcastic', 'negative', 'ironic']

# Emoji used for visual display of each sentiment category
EMOJIS = {
    'neutral': '😐',
    'positive': '😊',
    'mixed': '🤔',
    'sarcastic': '🙃',
    'negative': '😞',
    'ironic': '😏'
}

# Hex colours for styling based on sentiment category
COLOURS = {
    'neutral': '#D3D3D3',   # Light grey
    'positive': '#90EE90',  # Light green
    'mixed': '#FFD700',     # Gold
    'sarcastic': '#87CEEB', # Sky blue
    'negative': '#FFB6C1',  # Light pink
    'ironic': '#D8BFD8'     # Thistle purple
}

# Explanation messages displayed for each predicted sentiment
REVIEW_MESSAGES = {
    "positive": "✔️ This review is positive and expresses satisfaction.",
    "negative": "❌ This review is negative and shows dissatisfaction.",
    "neutral": "➖ This review is neutral and doesn’t show strong emotions.",
    "mixed": "🔀 This review contains both positive and negative feelings.",
    "sarcastic": "🙃 This review sounds sarcastic — it might mean the opposite of what it says.",
    "ironic": "😏 This review has an ironic tone — likely saying one thing but implying another."
}

# -=-=-=- Model Thresholds for Label Activation -=-=-=-
# Thresholds for converting model output probabilities to label predictions
custom_thresholds = {
    'neutral': 0.39,
    'positive': 0.509,
    'mixed': 0.22,
    'sarcastic': 0.29,
    'negative': 0.428,
    'ironic': 0.16
}

# -=-=-=- Sample Review Examples for Random Testing -=-=-=-
# List of pre-written review samples to test the app quickly
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
    "Good build, but the software is a nightmare.",
    "I’m happy with my purchase.",
    "It broke after two days. Useless.",
    "Nothing special, it’s just okay.",
    "Nice design, awful performance.",
    "Oh sure, because *that* feature really helped... not.",
    "This thing really nailed the 'barely works' vibe."
]

# -=-=-=- Streamlit UI Setup -=-=-=-
# Page title and instructions
st.title("💬 Sentiment Analysis")
st.write("Enter text or upload a file to get sentiment predictions.")

# -=-=-=- Session State Tracking and Initalise -=-=-=-
# Session state variables store values between Streamlit reruns
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "analyse_clicked" not in st.session_state:
    st.session_state.analyse_clicked = False
if "review_count" not in st.session_state:
    st.session_state.review_count = 0
if "file_key" not in st.session_state:
    st.session_state.file_key = 0

# -=-=-=- User Inputs (Text and File Upload) -=-=-=-
# Manual text input area
user_input = st.text_area("Enter text here:", key="input_text")

# File upload input, key is dynamic so it can be reset by changing file_key
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"], key=f"file_uploader_{st.session_state.file_key}")

# -=-=-=- Buttons (Analyse / Clear / Sample Review) -=-=-=-
st.markdown("###")
col1, col2, col3 = st.columns(3)

# Trigger sentiment analysis manually
with col1:
    if st.button("🚀 Analyse Sentiment"):
        st.session_state.analyse_clicked = True

# Clear/reset all input fields, predictions, and counters
with col2:
    if st.button("🗑️ Clear"):
        for key in list(st.session_state.keys()):
            if key not in ["shuffled_samples", "file_key"]:
                del st.session_state[key]
        st.session_state.file_key += 1  # Triggers a new key for the file uploader
        st.session_state.review_count = 0
        st.rerun()

# Load a random sample review from the list, shuffled with no repeats
with col3:
    if "shuffled_samples" not in st.session_state or not st.session_state.shuffled_samples:
        st.session_state.shuffled_samples = random.sample(SAMPLE_REVIEWS, len(SAMPLE_REVIEWS))

    if st.button("🎲 Try Sample Review"):
        for key in list(st.session_state.keys()):
            if key not in ["shuffled_samples", "file_key"]:
                del st.session_state[key]
        new_sample = st.session_state.shuffled_samples.pop()
        st.session_state.input_text = new_sample
        st.session_state.analyse_clicked = True
        st.rerun()

# -=-=-=- Sentiment Analysis Function -=-=-=-
# Runs inference on the input text using the model and applies thresholds

def analyse_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    predicted_labels = [label for label, prob in zip(LABELS, probs) if prob >= custom_thresholds[label]]

    # Convert to 'mixed' if both 'positive' and 'negative' are present
    if "positive" in predicted_labels and "negative" in predicted_labels and "mixed" not in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]
        predicted_labels.append("mixed")

    # Remove positive/negative if 'mixed' already exists
    if "mixed" in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]

    return predicted_labels if predicted_labels else ["neutral"]

# -=-=-=- Analyse Text Input (Single Review) -=-=-=-
# This section handles sentiment analysis for user-entered or sample review
if st.session_state.analyse_clicked:
    if st.session_state.input_text:
        with st.spinner("🔍 Analysing..."):
            user_input_single_line = " ".join(st.session_state.input_text.splitlines())

            # Show original input text
            st.markdown(
                f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review Entered: "{user_input_single_line}"</div>',
                unsafe_allow_html=True
            )

            # Analyse and display predictions
            sentiment = analyse_sentiment(user_input_single_line)
            sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
            sentiment_colour = COLOURS[sentiment[0]]

            st.markdown(
                f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:15px;">Sentiment: {sentiment_with_emojis}</div>',
                unsafe_allow_html=True
            )

            # Show explanation for each predicted sentiment
            for label in sentiment:
                st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)

            st.markdown("---")
    st.session_state.analyse_clicked = False

# -=-=-=- Analyse Uploaded File (Multi-Review) -=-=-=-
# Process every review in the uploaded .txt file, line-by-line
if uploaded_file:
    st.session_state.review_count = 0
    sentences = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
    st.subheader(f"Processing {len(sentences)} reviews from file...")

    for idx, sentence in enumerate(sentences, start=1):
        st.session_state.review_count += 1
        sentiment = analyse_sentiment(sentence)
        sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
        sentiment_colour = COLOURS[sentiment[0]]

        # Display original review with its index
        st.markdown(
            f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review #{st.session_state.review_count}: "{sentence}"</div>',
            unsafe_allow_html=True
        )
        # Show predicted sentiment block with coloured background
        st.markdown(
            f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:10px;">Sentiment: {sentiment_with_emojis}</div>',
            unsafe_allow_html=True
        )
        # Explanation for each detected sentiment
        for label in sentiment:
            st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)
        st.markdown("---")
