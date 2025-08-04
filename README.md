# Sentiment Analysis App

Aplikasi analisis sentimen berbasis Streamlit yang memanfaatkan model **TF-IDF** dan **Random Forest** untuk memprediksi sentimen dari review pengguna.

## Fitur

- Input teks review melalui web
- Prediksi sentimen: Positif 😊, Netral 😐, atau Negatif 😞
- Menampilkan confidence score dan probabilitas prediksi
- Tips input agar hasil lebih akurat

## Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/wildanmujjahid29/METLIT-Model-Deployment.git
   cd METLIT-Model-Deployment
   ```
2. **Buat virtual environment & install dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

## Struktur File

- `app.py` : Source code aplikasi Streamlit
- `random_forest_model.pkl` : Model Random Forest hasil training
- `tfidf_vectorizer.pkl` : Vectorizer TF-IDF hasil training
- `requirements.txt` : Daftar dependencies
- `.gitignore` : File/folder yang diabaikan git

## Catatan

- Model dan vectorizer sudah disiapkan, pastikan file `.pkl` tersedia di folder utama.
- Untuk deployment di platform cloud, pastikan environment sudah sesuai dengan `requirements.txt`.

---

Built with Streamlit · Model: TF-IDF + Random Forest
