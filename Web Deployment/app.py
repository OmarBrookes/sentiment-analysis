import streamlit as st
import torch
import transformers
import random
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# -=-=-=- Setup: Load model, tokenizer and set device -=-=-=-
# Load the fine-tuned RoBERTa model and tokenizer from Hugging Face
MODEL_PATH = "OmarBrookes/sentiment-analysis"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# Choose CUDA (GPU) if available for faster computation, fallback to CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -=-=-=- Configuration: Sentiment Labels, Emojis, Colours, Descriptions -=-=-=-
# The possible labels the model can predict
LABELS = ['neutral', 'positive', 'mixed', 'sarcastic', 'negative', 'ironic']

# Emojis displayed alongside predictions for visual feedback
EMOJIS = {
    'neutral': '😐',
    'positive': '😊',
    'mixed': '🤔',
    'sarcastic': '🙃',
    'negative': '😞',
    'ironic': '😏'
}

# Background colours for sentiment prediction blocks
COLOURS = {
    'neutral': '#D3D3D3',   # Light grey
    'positive': '#90EE90',  # Light green
    'mixed': '#FFD700',     # Gold
    'sarcastic': '#87CEEB', # Sky blue
    'negative': '#FFB6C1',  # Light pink
    'ironic': '#D8BFD8'     # Thistle purple
}

# Human-readable descriptions for each sentiment
REVIEW_MESSAGES = {
    "positive": "✔️ This review is positive and expresses satisfaction.",
    "negative": "❌ This review is negative and shows dissatisfaction.",
    "neutral": "➖ This review is neutral and doesn’t show strong emotions.",
    "mixed": "🔀 This review contains both positive and negative feelings.",
    "sarcastic": "🙃 This review sounds sarcastic — it might mean the opposite of what it says.",
    "ironic": "😏 This review has an ironic tone — likely saying one thing but implying another."
}

# -=-=-=- Custom Thresholds for Model Outputs -=-=-=-
# The model outputs probabilities (0 to 1). These thresholds decide when a label should be assigned.
custom_thresholds = {
    'neutral': 0.39,
    'positive': 0.509,
    'mixed': 0.22,
    'sarcastic': 0.29,
    'negative': 0.428,
    'ironic': 0.16
}

# -=-=-=- Sample Review Examples for Random Testing -=-=-=-
# This list is used when the user clicks the "Try Sample Review" button
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

# -=-=-=- Apply Custom Styling to Reduce Whitespace and Polish UI -=-=-=-
# This CSS tweak makes the app layout more compact and buttons cleaner
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        textarea {
            font-size: 16px !important;
        }
        .stButton button {
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# -=-=-=- Title and Introduction Text -=-=-=-
# Displayed at the top of the app
st.markdown("""
    <h2 style='text-align: center;'>💬 Sentiment Analysis</h2>
    <p style='text-align: center;'>Analyse the sentiment of reviews. Paste your review, upload a file, or try a sample.</p>
    <hr style='margin-top: 10px; margin-bottom: 25px;'>
""", unsafe_allow_html=True)

# -=-=-=- Session State Setup -=-=-=-
# These keep track of what the user is doing even after reruns
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "analyse_clicked" not in st.session_state:
    st.session_state.analyse_clicked = False
if "review_count" not in st.session_state:
    st.session_state.review_count = 0
if "file_key" not in st.session_state:
    st.session_state.file_key = 0

# -=-=-=- Input Fields: Text and File Upload -=-=-=-
# Users can either manually type in a review or upload a .txt file with reviews
st.markdown("#### Write or Upload a Review")
user_input = st.text_area("Type or paste your review here:", key="input_text")

# File uploader accepts plain text files with one review per line
uploaded_file = st.file_uploader("Or upload a .txt file with one review per line:", type=["txt"], key=f"file_uploader_{st.session_state.file_key}")

# -=-=-=- Control Buttons: Analyse / Clear / Sample -=-=-=-
# Three interactive buttons neatly centered below the input
st.markdown("<div style='margin-top: 20px; text-align: center;'>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 1, 1])

# Button 1: Analyse sentiment of user input
with col_a:
    if st.button("🚀 Analyse Sentiment"):
        st.session_state.analyse_clicked = True

# Button 2: Clear/reset all state and inputs
with col_b:
    if st.button("🗑️ Clear"):
        for key in list(st.session_state.keys()):
            if key not in ["shuffled_samples", "file_key"]:
                del st.session_state[key]
        st.session_state.file_key += 1  # Triggers Streamlit to refresh file uploader
        st.session_state.review_count = 0
        st.rerun()

# Button 3: Randomly load one of the pre-defined sample reviews
with col_c:
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

st.markdown("</div>", unsafe_allow_html=True)

# -=-=-=- Sentiment Prediction Logic -=-=-=-
# This function runs the model on the text and applies custom thresholds

def analyse_sentiment(text):
    # Tokenise and prepare the input for the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference without updating gradients
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    # Convert probabilities to labels based on thresholds
    predicted_labels = [label for label, prob in zip(LABELS, probs) if prob >= custom_thresholds[label]]

    # Custom rule: if both positive and negative exist, call it 'mixed'
    if "positive" in predicted_labels and "negative" in predicted_labels and "mixed" not in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]
        predicted_labels.append("mixed")

    # If mixed is already present, remove separate positive/negative
    if "mixed" in predicted_labels:
        predicted_labels = [label for label in predicted_labels if label not in ["positive", "negative"]]

    # Always return at least one label, default is 'neutral'
    return predicted_labels if predicted_labels else ["neutral"]

# -=-=-=- Handle Analysis for Typed or Sample Input -=-=-=-
if st.session_state.analyse_clicked:
    if st.session_state.input_text:
        with st.spinner("Analysing sentiment..."):
            user_input_single_line = " ".join(st.session_state.input_text.splitlines())

            # Show what the user entered
            st.markdown(f'<div style="padding:10px 0;font-weight:bold;">📝 Review Entered:</div><div style="margin-bottom:5px">"{user_input_single_line}"</div>', unsafe_allow_html=True)

            # Run model and show results with background colour
            sentiment = analyse_sentiment(user_input_single_line)
            sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
            sentiment_colour = COLOURS[sentiment[0]]

            st.markdown(
                f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:8px;color:black;font-weight:bold;margin-bottom:15px;">Sentiment: {sentiment_with_emojis}</div>',
                unsafe_allow_html=True
            )

            # Display explanation text for each predicted sentiment
            for label in sentiment:
                st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin-top:30px;'>", unsafe_allow_html=True)
    st.session_state.analyse_clicked = False

# -=-=-=- Handle File Upload Sentiment Analysis -=-=-=-
if uploaded_file:
    # Reset count for numbering file reviews
    st.session_state.review_count = 0
    sentences = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
    st.subheader(f"📂 Processing {len(sentences)} reviews from uploaded file")

    for idx, sentence in enumerate(sentences, start=1):
        st.session_state.review_count += 1
        sentiment = analyse_sentiment(sentence)
        sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
        sentiment_colour = COLOURS[sentiment[0]]

        # Show review number and original text
        st.markdown(
            f'<div style="padding:10px 0;font-weight:bold;">📝 Review #{st.session_state.review_count}:</div><div style="margin-bottom:5px">"{sentence}"</div>',
            unsafe_allow_html=True
        )
        # Sentiment box with colour
        st.markdown(
            f'<div style="background-color:{sentiment_colour};padding:10px;border-radius:8px;color:black;font-weight:bold;margin-bottom:10px;">Sentiment: {sentiment_with_emojis}</div>',
            unsafe_allow_html=True
        )
        # Label explanations
        for label in sentiment:
            st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top:20px;'>", unsafe_allow_html=True)