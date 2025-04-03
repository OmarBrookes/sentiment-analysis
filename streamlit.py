import streamlit as st
import torch
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# -=-=-=- Setup: Load model, tokenizer and set device -=-=-=-
MODEL_PATH = "OmarBrookes/sentiment-analysis"  # The model’s Hugging Face path 
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# Automatically detect GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label names, corresponding emojis, and background colours
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
    'neutral': '#D3D3D3',   # Light grey
    'positive': '#90EE90',   # Light green
    'mixed': '#FFD700',      # Gold
    'sarcastic': '#87CEEB',  # Sky blue
    'negative': '#FFB6C1',   # Light pink
    'ironic': '#D8BFD8'      # Purple (Thistle)
}

REVIEW_MESSAGES = {
    "positive": "✔️ This review is positive and expresses satisfaction.",
    "negative": "❌ This review is negative and shows dissatisfaction.",
    "neutral": "➖ This review is neutral and doesn’t show strong emotions.",
    "mixed": "🔀 This review contains both positive and negative feelings.",
    "sarcastic": "🙃 This review sounds sarcastic — it might mean the opposite of what it says.",
    "ironic": "😏 This review has an ironic tone — likely saying one thing but implying another."
}

# -=-=-=- Custom Thresholds -=-=-=-
# These thresholds have been adjusted to fine-tune label detection
custom_thresholds = {
    'neutral': 0.39,
    'positive': 0.509,
    'mixed': 0.22,
    'sarcastic': 0.29,
    'negative': 0.428,
    'ironic': 0.16
}

# -=-=-=- Streamlit App Interface -=-=-=-
st.title("💬 Sentiment Analysis")
st.write("Enter text or upload a file to get sentiment predictions.")

user_input = st.text_area("Enter text here:")
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
review_count = 0 # To keep track of the review numbers

# -=-=-=- Prediction Function Using Custom Thresholds -=-=-=-
def analyse_sentiment(text):
    # Tokenize the input text, with truncation and padding up to 128 tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply sigmoid to get probabilities from logits
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    
    # Compare each probability against the custom threshold for that label
    predicted_labels = [label for label, prob in zip(LABELS, probs) if prob >= custom_thresholds[label]]
    
    # If no label meets the threshold, return 'neutral' by default
    return predicted_labels if predicted_labels else ["neutral"]

# -=-=-=- Interactive Prediction -=-=-=-
if st.button("Analyse Sentiment"):
    if user_input:
        review_count += 1
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
        
        # Display review interpretation
        for label in sentiment:
            st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)

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
                # Display review interpretation
                for label in sentiment:
                    st.markdown(f"<div style='margin-bottom:10px;'>{REVIEW_MESSAGES[label]}</div>", unsafe_allow_html=True)
