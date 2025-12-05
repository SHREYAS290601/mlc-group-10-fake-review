import nltk
import ssl

# 1. Bypass SSL verification (Mac issue workaround)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Download the data
print("Downloading stopwords...")
nltk.download('stopwords')
print("✅ Download complete.")