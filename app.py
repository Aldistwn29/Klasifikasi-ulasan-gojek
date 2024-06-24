import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
import nltk
from nlp_id.lemmatizer import Lemmatizer
from indoNLP.preprocessing import replace_slang, replace_word_elongation
from wordcloud import WordCloud, STOPWORDS
from xgboost import XGBClassifier

# Fungsi untuk mengunduh stopword bahasa Indonesia
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('indonesian')

# Setup halaman
st.set_page_config(page_title="Analisis Sentimen Ulasan Gojek", layout="wide")

# Title
st.title("Analisis Sentimen Ulasan Gojek")

# Fungsi untuk memuat dan memproses data
@st.cache(ttl=3600)  
def load_data():
    df = pd.read_csv('gojek_review.csv')
    df = df.drop(columns=['userName'])
    df = df[df['score'] != 3]
    target_map = {1: "negative", 2: "negative", 4: "positive", 5: "positive"}
    df['target'] = df['score'].map(target_map)
    target_map_biner = {"negative": 0, "positive": 1}
    df['target_biner'] = df['target'].map(target_map_biner)
    df.columns = ['text', 'score', 'date', 'app_version', 'target', 'target_binery']
    df['app_version_cut'] = df['app_version'].str[:3]
    df = df[df['app_version_cut'] >= '4.7']
    
    lemmatizer = Lemmatizer()
    df['text_processed'] = df['text'].apply(lemmatizer.lemmatize)
    df['text_processed'] = df['text_processed'].apply(replace_slang)
    df['text_processed'] = df['text_processed'].apply(replace_word_elongation)
    
    def tokenize_text(text):
        text = text.strip()
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        return tokens

    def remove_stopword(tokens):
        tokens = [token for token in tokens if token not in stopwords]
        return tokens

    df['tokens'] = df['text_processed'].apply(tokenize_text)
    df['tokens_without_stopword'] = df['tokens'].apply(remove_stopword)
    df['text_processed'] = df['tokens_without_stopword'].apply(" ".join)
    
    return df

# Load data
df = load_data()

# Menampilkan data
st.subheader("Data Ulasan Gojek")
st.write(df.head())

# Visualisasi countplot dengan judul
st.subheader("Visualisasi Distribusi Sentimen berdasarkan Versi Aplikasi")
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, hue='app_version_cut')
plt.title("Distribusi Sentimen berdasarkan Versi Aplikasi")
st.pyplot(fig)

# Visualisasi distribusi sentimen
st.subheader("Distribusi Sentimen")
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax, palette='pastel')
plt.title("Distribusi Sentimen")
st.pyplot(fig)

# Wordcloud
st.subheader("Wordcloud Sentimen")
sentiment_list = ['negative', 'positive']
colormap_list = ['Reds_r', 'Blues_r']
fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
stopwords_set = set(STOPWORDS)

for i, (sentiment, colormap) in enumerate(zip(sentiment_list, colormap_list)):
    text = ' '.join(text for text in df[df['target'] == sentiment]['text_processed'])
    wc = WordCloud(colormap=colormap, stopwords=stopwords_set, width=1600, height=900).generate(text)
    ax[i].imshow(wc)
    ax[i].set_title(sentiment + " wordcloud", fontsize=18)
    ax[i].axis('off')

fig.tight_layout()
st.pyplot(fig)

# Model training
st.subheader("Model Training")
X = df['text_processed']
y = df['target_binery']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

@st.cache(ttl=3600)  # Cache data for 3600 seconds (1 hour)
def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return vectorizer, X_train, X_test

vectorizer, X_train, X_test = vectorize_data(X_train, X_test)

# Model Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)


train_acc = log.score(X_train, y_train)
test_acc = log.score(X_test, y_test)
train_auc = roc_auc_score(y_train, log.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, log.predict_proba(X_test)[:, 1])

st.subheader("Training Model Logistic Regression")
st.write(f"Train Accuracy: {train_acc}")
st.write(f"Test Accuracy: {test_acc}")
st.write(f"Train AUC: {train_auc}")
st.write(f"Test AUC: {test_auc}")

# Model Xboost 
xg = XGBClassifier(objective='binary:logistic',max_depth=8, n_estimators=400, learning_rate=0.3)
xg.fit(X_train, y_train)

pr_train = xg.predict_proba(X_train)[:,1]
pr_test = xg.predict_proba(X_test)[:,1]

st.subheader("Training Model Xboost")
st.write("Train Accuracy: ", xg.score(X_train, y_train))
st.write("Test Accuracy: ", xg.score(X_test, y_test))
st.write("Train AUC: ", roc_auc_score(y_train, pr_train))
st.write("Test AUC: ", roc_auc_score(y_test, pr_test))


# Model evaluation
st.subheader("Evaluasi Model")
def report(model):
    preds = model.predict(X_test)
    st.write(classification_report(y_test, preds, target_names=['negative', 'positive']))
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

st.subheader("Evaluasi Model Logistic Regression")
report(log)

st.subheader("Evaluasi Model Xboost")
report(xg)

# Prediksi Sentimen Teks
st.subheader("Prediksi Sentimen Teks")

# Menambahkan judul di atas form input
st.markdown("### Masukkan teks untuk memprediksi apakah sentimen positive atau negative")

user_input = st.text_area("Masukkan teks ulasan Anda:")

# Tombol kirim
if st.button("Kirim"):
    # Praproses teks masukan pengguna
    @st.cache(ttl=3600)  # Cache data for 3600 seconds (1 hour)
    def preprocess_text(text):
        lemmatizer = Lemmatizer()
        text = lemmatizer.lemmatize(text)
        text = replace_slang(text)
        text = replace_word_elongation(text)
        
        def tokenize_text(text):
            text = text.strip()
            tokens = nltk.word_tokenize(text)
            tokens = [token for token in tokens if token.isalpha()]
            return tokens

        def remove_stopword(tokens):
            tokens = [token for token in tokens if token not in stopwords]
            return tokens

        tokens = tokenize_text(text)
        tokens = remove_stopword(tokens)
        processed_text = " ".join(tokens)
        return processed_text

    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])

    # Prediksi menggunakan model Logistic Regression
    prediction = log.predict(input_vector)
    prediction_proba = log.predict_proba(input_vector)

    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        st.write("Prediksi Sentimen: positive")
    else:
        st.write("Prediksi Sentimen: negative")

    st.write(f"Probabilitas Sentimen positive: {prediction_proba[0][1]:.2f}")
    st.write(f"Probabilitas Sentimen negative: {prediction_proba[0][0]:.2f}")