
# Data Analyst Local Stack

## 📊 Tentang Proyek
Dashboard analitik e-commerce dengan stack open-source 100% lokal.

## 🛠️ Tools Used
- **Database**: PostgreSQL
- **Analytics**: Python, Pandas, Jupyter
- **Machine Learning**: Scikit-learn, Statsmodels, Prophet
- **AI Assistant**: Ollama + Gemma2/Mistral
- **Dashboard**: Streamlit + Plotly

## 🚀 Cara Menjalankan
1. Aktifkan environment: `source dataanalyst_env/bin/activate`
2. Jalankan PostgreSQL: `brew services start postgresql`
3. Jalankan Ollama: `ollama serve`
4. Jalankan dashboard: `streamlit run dashboard/streamlit_app.py`

## 📁 Struktur Proyek
- `/data`: Data mentah dan processed
- `/notebooks`: Jupyter notebooks untuk eksplorasi
- `/scripts`: Script Python utilities
- `/dashboard`: Streamlit dashboard
- `/sql`: SQL queries

## ✨ Fitur
- Dashboard interaktif real-time
- Customer segmentation (RFM)
- Sales forecasting dengan ARIMA/Prophet
- AI Assistant dengan local LLM
- Export data ke CSV
