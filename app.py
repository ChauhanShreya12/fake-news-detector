import streamlit as st
import numpy as np
import re
import pandas as pd
import os
import urllib.request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords', quiet=True)

# ── Auto-download real dataset if sample data is detected ─────────────────────
def get_real_data():
    csv_path = 'train.csv'
    if os.path.exists(csv_path):
        check = pd.read_csv(csv_path, nrows=3)
        text_val = str(check['text'].values if 'text' in check.columns else '')
        if 'fabricated' not in text_val and 'factual news article based' not in text_val:
            return  # already real data, do nothing
    # Try downloading real dataset
    for url in [
        "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv",
        "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
    ]:
        try:
            urllib.request.urlretrieve(url, "real_news_temp.csv")
            raw = pd.read_csv("real_news_temp.csv").fillna('')
            if 'label' in raw.columns and 'title' in raw.columns:
                raw['author'] = 'Unknown'
                if raw['label'].dtype == object:
                    raw['label'] = raw['label'].map({'FAKE': 1, 'REAL': 0})
                raw['id'] = range(len(raw))
                raw[['id','title','author','text','label']].dropna().to_csv(csv_path, index=False)
            break
        except Exception:
            continue

get_real_data()

# ── Original code — all variable names unchanged ───────────────────────────────
# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
news_df['label'] = news_df['label'].map({'FAKE': 1, 'REAL': 0}).astype(int)
y = news_df['label']
# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)

# ── Website ────────────────────────────────────────────────────────────────────
st.title('Fake News Detector')

# Show accuracy
training_accuracy = accuracy_score(Y_train, model.predict(X_train))
testing_accuracy  = accuracy_score(Y_test,  model.predict(X_test))
st.success(f"✅ Model ready! Train accuracy: {training_accuracy*100:.2f}%  |  Test accuracy: {testing_accuracy*100:.2f}%")

st.markdown("---")

# ── Original prediction function — unchanged ──────────────────────────────────
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')

st.markdown("---")

# ── Bonus: example buttons to test real and fake news ─────────────────────────
st.subheader("📋 Click any example to test instantly")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔴 Fake news examples — click to test:**")
    for ex in [
        "SHOCKING: Government hiding truth about vaccines and microchips",
        "Scientists CONFIRM moon landing was staged and completely fake",
        "Obama secretly Muslim born in Kenya new documents confirm",
        "5G towers spreading virus WHO whistleblower exposes secret",
    ]:
        if st.button(ex, key="f_"+ex[:20]):
            pred = prediction(ex)
            if pred == 1:
                st.error("🔴 The News is Fake")
            else:
                st.success("🟢 The News Is Real")

with col2:
    st.markdown("**🟢 Real news examples — click to test:**")
    for ex in [
        "Federal Reserve raises interest rates by 25 basis points",
        "NASA confirms successful launch of new Mars exploration rover",
        "WHO releases updated guidelines on antibiotic resistance",
        "Supreme Court issues ruling on landmark civil rights case",
    ]:
        if st.button(ex, key="r_"+ex[:20]):
            pred = prediction(ex)
            if pred == 1:
                st.error("🔴 The News is Fake")
            else:
                st.success("🟢 The News Is Real")
