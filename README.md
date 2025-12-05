# Gatekeeper: Fake Review Detector

Gatekeeper is an advanced fake review detection system built with Streamlit. It leverages a Hybrid Bi-LSTM neural network combined with Transformer-based forensics (Google's Electra) to analyze product reviews and determine their authenticity.

##  Features

- **Hybrid Deep Learning Model:** Combines text sequences (processed via Bi-LSTM) with engineered numerical features (Sentiment, Readability, POS tags) for robust classification.
- **Forensic Analysis:** Uses the Electra Discriminator model to identify and highlight suspicious tokens within the text.
- **Linguistic Feature Extraction:** Automatically extracts and analyzes:
  - Sentiment Polarity & Subjectivity
  - Readability Scores (Flesch-Kincaid)
  - Part-of-Speech (POS) Ratios (Nouns, Verbs, Adjectives)
  - Entity Counts & Capitalization Ratios
- **Hardware Acceleration:** Automatically detects and utilizes available hardware acceleration (Apple Silicon MPS, NVIDIA CUDA, or CPU).
- **Interactive Dashboard:** Real-time analysis with adjustable sensitivity thresholds and detailed metric visualization.

##  Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The `requirements.txt` includes `tensorflow-macos` and `tensorflow-metal` for Apple Silicon optimization. If you are on Windows or Linux, you may need to replace these with standard `tensorflow`.*

3. **Download Spacy Model (Optional):**
   The application attempts to download the required Spacy model automatically on first run, but you can pre-install it:
   ```bash
   python -m spacy download en_core_web_sm
   ```

##  Project Structure

- `main.py`: The main Streamlit application entry point.
- `artifacts/`: Directory containing trained models and pre-processing tools.
  - `hybrid_lstm_model.h5`: The trained Keras Bi-LSTM model.
  - `feature_scaler.joblib`: Scaler for numerical features.
  - `label_encoder.joblib`: Encoder for target labels.
  - `tokenizer.pickle`: Keras tokenizer for text sequences.
  - `tfidf_vectorizer.joblib`: (Optional) TF-IDF vectorizer.
- `requirements.txt`: List of Python dependencies.

##  Usage

1. **Ensure artifacts are present:**
   Make sure all required model files are located in the `artifacts/` directory.

2. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

3. **Analyze Reviews:**
   - Open the provided local URL in your browser.
   - Paste a product review into the text area.
   - Click ** Analyze Review**.
   - View the prediction (Real/Fake), confidence score, and forensic details.

##  Configuration

Use the sidebar to adjust the **Forensic Sensitivity** slider. A lower threshold results in stricter detection of suspicious tokens. You can also toggle the display of raw feature data and detailed token scores.

##  Models Used

- **Bi-LSTM (Bidirectional Long Short-Term Memory):** For sequence classification of review text.
- **Electra-Small-Discriminator:** For token-level anomaly detection (identifying "generated" or suspicious words).
- **Spacy & NLTK:** For NLP preprocessing and linguistic feature extraction.
