import streamlit as st
import pandas as pd
import numpy as np
import joblib
import string
import time
import os
import re
import torch
import tensorflow as tf
import spacy
import textstat
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from scipy.sparse import hstack, csr_matrix
from transformers import ElectraTokenizer, AutoModelForPreTraining

# --- HARDWARE ACCELERATION SETUP ---
try:
    if torch.backends.mps.is_available():
        TORCH_DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        TORCH_DEVICE = torch.device("cuda")
    else:
        TORCH_DEVICE = torch.device("cpu")
except Exception:
    TORCH_DEVICE = torch.device("cpu")

try:
    # Initialize TensorFlow GPU connection
    _ = tf.constant([1.0])
except:
    pass

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Gatekeeper: Fake Review Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetricLabel {
        font-weight: bold;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)


# --- EAGER LOADER ---
@st.cache_resource
def load_all_models_and_tools():
    print("--- STARTING EAGER LOAD ---")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    negation_words = {'no', 'not', 'nor', 'never', "n't", "don't", "aren't", "couldn't", "didn't", "isn't", "wasn't", "won't", "wouldn't"}
    final_stopwords = stop_words - negation_words

    e_tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    e_model = AutoModelForPreTraining.from_pretrained("google/electra-small-discriminator")
    e_model.to(TORCH_DEVICE) 
    e_model.eval()

    artifacts = {}
    try:
        artifacts['tfidf'] = joblib.load('artifacts/tfidf_vectorizer.joblib')
        artifacts['scaler'] = joblib.load('artifacts/feature_scaler.joblib')
        artifacts['le'] = joblib.load('artifacts/label_encoder.joblib')
        artifacts['lstm_model'] = load_model('artifacts/hybrid_lstm_model.h5')
        
        import pickle
        with open('artifacts/tokenizer.pickle', 'rb') as handle:
            artifacts['keras_tokenizer'] = pickle.load(handle)
    except FileNotFoundError:
        artifacts = None

    return nlp, final_stopwords, negation_words, e_tokenizer, e_model, artifacts

# --- INITIALIZE ---
with st.spinner("⚡ Waking up GPU & Loading AI Models..."):
    nlp, final_stopwords, negation_words, electra_tokenizer, electra_model, artifacts = load_all_models_and_tools()

# --- PROCESSING FUNCTIONS ---
def clean_text_robust(text):
    text = str(text).lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        is_negation = token.text in negation_words
        if (not token.is_punct and not token.is_space) and (is_negation or not token.is_stop):
            tokens.append(token.lemma_)
    return " ".join(tokens)

def get_advanced_features(text):
    blob = TextBlob(text)
    doc = nlp(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    fk_score = textstat.flesch_kincaid_grade(text)
    pos_counts = Counter([token.pos_ for token in doc])
    entity_count = len(doc.ents)
    char_count = len(text)
    caps_count = sum(1 for c in text if c.isupper())
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    return {
        'sentiment_polarity': polarity,
        'sentiment_subjectivity': subjectivity,
        'readability_fk': fk_score,
        'noun_count': pos_counts.get('NOUN', 0),
        'verb_count': pos_counts.get('VERB', 0),
        'adj_count': pos_counts.get('ADJ', 0),
        'entity_count': entity_count,
        'caps_ratio': caps_count / char_count if char_count > 0 else 0,
        'punct_count': exclamation_count + question_count
    }

def analyze_tokens_electra_gpu(text, threshold=0.12):
    if electra_model is None: return []
    
    inputs = electra_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = electra_model(**inputs)
        
    probs = torch.sigmoid(outputs.logits).squeeze(0).cpu().tolist()
    tokens = electra_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    
    suspicious = []
    for tok, prob in zip(tokens, probs):
        clean_tok = tok.replace("##", "")
        if clean_tok.lower() in ("[cls]", "[sep]", "[pad]", "[unk]"): continue
        if all(ch in string.punctuation for ch in clean_tok): continue
        if prob > threshold:
            suspicious.append((clean_tok, float(prob)))
            
    merged = {}
    for t, p in suspicious:
        merged[t] = max(merged.get(t, 0.0), p)
        
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)

# --- UI LAYOUT ---
st.title("⚡ Gatekeeper")
st.subheader("Advanced Fake Review Detection System")

with st.sidebar:
    st.header("⚙️ Configuration")
    threshold = st.slider("Forensic Sensitivity", 0.05, 0.30, 0.12, 0.01, help="Lower threshold = stricter detection")
    show_raw_features = st.checkbox("Show Raw Features", value=False)
    show_token_scores = st.checkbox("Show Detailed Token Scores", value=True)

    st.markdown("---")
    st.markdown(f"**Hardware:** {TORCH_DEVICE}")
    st.markdown("**Status:** Ready ✅")

# Main Input Area
col1, col2 = st.columns([2, 1])
with col1:
    review_text = st.text_area("Review Text:", height=200, placeholder="Paste review content here to analyze...")
with col2:
    st.info("💡 **Tip:** Paste a product review to detect authenticity. The system uses a Hybrid Bi-LSTM neural network + Transformer forensics.")
    analyze_btn = st.button("🚀 Analyze Review", type="primary", use_container_width=True)

if analyze_btn:
    if not review_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # --- ANALYSIS PIPELINE ---
        start_total = time.time()
        
        with st.status("Processing Review...", expanded=True) as status:
            st.write("Cleaning text & extracting linguistic features...")
            cleaned_text = clean_text_robust(review_text)
            raw_feats = get_advanced_features(review_text)
            wc = len(review_text.split())
            wc_safe = wc if wc > 0 else 1
            
            num_data = {
                'raw_word_count': wc,
                'sentiment_polarity': raw_feats['sentiment_polarity'],
                'sentiment_subjectivity': raw_feats['sentiment_subjectivity'], 
                'readability_fk': raw_feats['readability_fk'],
                'noun_ratio': raw_feats['noun_count'] / wc_safe,
                'verb_ratio': raw_feats['verb_count'] / wc_safe,
                'adj_ratio': raw_feats['adj_count'] / wc_safe, 
                'entity_ratio': raw_feats['entity_count'] / wc_safe, 
                'caps_ratio': raw_feats['caps_ratio'], 
                'punct_count': raw_feats['punct_count']
            }
            
            numeric_cols = [
                'raw_word_count', 'sentiment_polarity', 'sentiment_subjectivity', 
                'readability_fk', 'noun_ratio', 'verb_ratio', 'adj_ratio', 
                'entity_ratio', 'caps_ratio', 'punct_count'
            ]
            
            input_df = pd.DataFrame([num_data])[numeric_cols]
            
            st.write("Running Neural Network Inference...")
            if artifacts:
                X_num = artifacts['scaler'].transform(input_df)
                seq = artifacts['keras_tokenizer'].texts_to_sequences([cleaned_text])
                X_seq = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
                pred_prob = artifacts['lstm_model'].predict([X_seq, X_num], verbose=0)[0][0]
                
                # if hasattr(artifacts['le'], 'classes_'):
                #     classes = artifacts['le'].classes_
                #     # Assuming predicted_label maps to class 1 or 0
                #     predicted_label = classes[1] if pred_prob > 0.5 else classes[0]
                #     confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
                # else:
                #     predicted_label = "Fake" if pred_prob > 0.5 else "Real"
                #     confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
                if hasattr(artifacts['le'], 'classes_'):
                    classes = artifacts['le'].classes_
                    
                    # DEBUG: Print mapping to terminal to confirm
                    # print(f"Class 0: {classes[0]}, Class 1: {classes[1]}") 

                    # Usually: 0=CG (Fake), 1=OR (Real)
                    # If model outputs prob of Class 1 (Real):
                    if pred_prob > 0.5:
                        predicted_label = classes[1] # Likely "OR" (Real)
                        confidence = pred_prob
                        # Invert logic for "Fake Probability" display if needed
                        is_fake = False
                    else:
                        predicted_label = classes[0] # Likely "CG" (Fake)
                        confidence = 1 - pred_prob
                        is_fake = True
                else:
                    # Fallback
                    predicted_label = "Fake" if pred_prob < 0.5 else "Real"
                    confidence = 1 - pred_prob if pred_prob < 0.5 else pred_prob
            else:
                pred_prob = 0.85
                predicted_label = "Demo Fake"
                confidence = 0.99

            st.write("Performing Forensic Analysis...")
            suspicious_tokens = analyze_tokens_electra_gpu(review_text, threshold)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
        
        end_total = time.time()
        latency = round((end_total - start_total) * 1000, 2)

        # --- METRICS DASHBOARD ---
        st.markdown("### 📊 Analysis Results")
        
        m1, m2, m3, m4 = st.columns(4)
        
        # Color coding
        pred_color = "red" if "Fake" in predicted_label or "CG" in predicted_label else "green"
        
        m1.markdown(f"**Prediction**<br><span style='color:{pred_color}; font-size: 24px; font-weight: bold'>{predicted_label}</span>", unsafe_allow_html=True)
        m2.metric("Confidence Score", f"{confidence:.2%}")
        m3.metric("Processing Time", f"{latency} ms")
        m4.metric("Suspicious Tokens", len(suspicious_tokens))
        
        st.progress(float(confidence), text=f"Model Certainty: {predicted_label}")

        # --- DETAILED METADATA ---
        if show_raw_features:
            with st.expander("📈 View Linguistic Metadata"):
                f_col1, f_col2, f_col3 = st.columns(3)
                f_col1.metric("Word Count", num_data['raw_word_count'])
                f_col1.metric("Sentiment Polarity", f"{num_data['sentiment_polarity']:.2f}")
                f_col2.metric("Readability (Grade)", f"{num_data['readability_fk']:.1f}")
                f_col2.metric("Subjectivity", f"{num_data['sentiment_subjectivity']:.2f}")
                f_col3.metric("Uppercase Ratio", f"{num_data['caps_ratio']:.2%}")
                f_col3.metric("Punctuation Count", num_data['punct_count'])

        # --- FORENSICS ---
        st.markdown("### 🕵️ Forensic Text Analysis")
        
        if suspicious_tokens:
            annotated_text = review_text
            
            # 1. Sort tokens by length (longest first) to avoid partial replacement issues
            suspicious_tokens_sorted = sorted(suspicious_tokens, key=lambda x: len(x[0]), reverse=True)
            placeholders = {}
            
            # 2. Replace suspicious words with safe placeholders first
            for i, (word, score) in enumerate(suspicious_tokens_sorted):
                placeholder = f"__IGTR_PH_{i}__"
                placeholders[placeholder] = (word, score)
                if word.isalnum():
                    # Match whole words only (case insensitive)
                    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                    annotated_text = pattern.sub(placeholder, annotated_text)
                else:
                    # Handle punctuation/symbols
                    annotated_text = annotated_text.replace(word, placeholder)

            # 3. Replace placeholders with HTML spans (Dynamic Alpha Redness)
            for placeholder, (word, score) in placeholders.items():
                # Calculate alpha based on score (min 0.2, max 0.8 for visibility)
                # score is usually between threshold (e.g. 0.15) and 1.0
                alpha = max(0.2, min(0.8, score))
                
                html_span = (
                    f"<span style='background-color: rgba(255, 0, 0, {alpha:.2f}); "
                    f"padding: 2px; border-radius: 3px; border-bottom: 1px solid red; "
                    f"cursor: help;' title='Artificial Probability: {score:.2%}'>{word}</span>"
                )
                annotated_text = annotated_text.replace(placeholder, html_span)

            st.markdown(
                f"""
                <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #e9ecef; line-height: 1.6; font-size: 16px; color: #212529;">
                    {annotated_text}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if show_token_scores:
                with st.expander("View Detailed Token Scores"):
                    token_df = pd.DataFrame(suspicious_tokens, columns=["Token", "Artificial Probability"])
                    # Store raw score for sorting/data, format for display
                    display_df = token_df.copy()
                    display_df["Artificial Probability"] = display_df["Artificial Probability"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(display_df, use_container_width=True)
        else:
            st.success("✅ No suspicious token patterns detected by forensic analysis.")
            st.markdown(f"<div style='padding:15px; background:#f0fdf4; border-radius:8px; color:#166534'>{review_text}</div>", unsafe_allow_html=True)