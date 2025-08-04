import joblib
import numpy as np
import streamlit as st

# Load model dan vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("random_forest_model.pkl")

st.title("Sentiment Analysis App")
st.markdown("""
Selamat datang di aplikasi analisis sentimen!<br>
Model ini menggunakan **TF-IDF** dan **Random Forest** untuk memprediksi sentimen dari review yang kamu masukkan.<br>
Sentimen yang diprediksi: <b>Positif</b>, <b>Netral</b>, atau <b>Negatif</b>.<br>
<br>
<i>Masukkan review pada kolom di bawah, lalu klik <b>Analyze</b> untuk melihat hasil prediksi.</i>
""", unsafe_allow_html=True)

# Input form
with st.form("prediction_form"):
    user_input = st.text_area("Enter a review:", height=150)
    submit = st.form_submit_button("Analyze")

if submit:
    if not user_input.strip():
        st.warning("Silakan masukkan teks review terlebih dahulu.")
    else:
        # Preprocess input
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        proba = model.predict_proba(input_tfidf)[0]

        # Mapping label & emoji
        label_map = {-1: "Negatif", 0: "Netral", 1: "Positif"}
        emoji_map = {-1: "😞", 0: "😐", 1: "😊"}
        sentiment = label_map.get(prediction, "Unknown")
        sentiment_emoji = emoji_map.get(prediction, "")

        # Display result
        st.subheader("Hasil Prediksi")
        st.markdown(f"**Sentimen:** {sentiment_emoji} `{sentiment}`")

        # Display confidence
        confidence = proba[prediction]
        st.markdown(f"**Confidence:** `{confidence:.2f}` _(Semakin tinggi, semakin yakin model terhadap prediksi)_")

        # Display probability table
        st.subheader("Probabilitas Prediksi")
        prob_dict = {
            "Negatif": proba[0],
            "Netral": proba[1],
            "Positif": proba[2]
        }
        st.table(prob_dict)

        # Tips input
        st.info("Tips: Masukkan review yang jelas dan detail agar hasil prediksi lebih akurat.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit · Model: TF-IDF + Random Forest · Created by [Your Name]")
