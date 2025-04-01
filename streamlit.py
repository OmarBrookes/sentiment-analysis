import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

MODEL_PATH = "OmarBrookes/my-sentiment-analysis"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABELS = ['neutral', 'positive', 'mixed', 'sarcastic', 'negative', 'ironic']
EMOJIS = {
    'neutral': '😐',
    'positive': '😊',
    'mixed': '🤔',
    'sarcastic': '🙃',
    'negative': '😞',
    'ironic': '😏'
}
COLORS = {
    'neutral': '#D3D3D3',  
    'positive': '#90EE90',  
    'mixed': '#FFD700',  
    'sarcastic': '#87CEEB', 
    'negative': '#FFB6C1',  
    'ironic': '#D8BFD8'  
}

st.title("Sentiment Analysis")
st.write("Enter a text or upload a file to get sentiment predictions.")

user_input = st.text_area("Enter your text here:")
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

review_count = 0

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    predicted_labels = [label for label, prob in zip(LABELS, probs) if prob >= 0.5]
    return predicted_labels if predicted_labels else ["neutral"]

if st.button("Analyze Sentiment"):
    if user_input:
        review_count += 1
        st.subheader(f"Review #{review_count}")
        st.markdown(f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review: "{user_input}"</div>', unsafe_allow_html=True)
        sentiment = analyze_sentiment(user_input)
        sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
        sentiment_color = COLORS[sentiment[0]]  
        st.markdown(f'<div style="background-color:{sentiment_color};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:15px;">Sentiment: {sentiment_with_emojis}</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        sentences = uploaded_file.read().decode("utf-8").splitlines()
        st.subheader(f"Processing {len(sentences)} reviews from file...")
        for idx, sentence in enumerate(sentences, start=1):
            if sentence.strip():  
                review_count += 1
                sentiment = analyze_sentiment(sentence)
                sentiment_with_emojis = ', '.join([f"{EMOJIS[label]} {label}" for label in sentiment])
                sentiment_color = COLORS[sentiment[0]]  
                st.markdown(f'<div style="padding:10px;margin-bottom:5px;font-weight:bold;">📝 Review #{review_count}:"{sentence}"</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color:{sentiment_color};padding:10px;border-radius:5px;color:black;font-weight:bold;margin-bottom:15px;">Sentiment: {sentiment_with_emojis}</div>', unsafe_allow_html=True)
