# 🚀 AI-Powered Data Analytics

### Local AI-Native Analytics Platform (PostgreSQL + Python + Local LLM)

<p align="center">
  <img src="https://github.com/user-attachments/assets/489788b6-0104-4011-8278-0a1612aee068"
       alt="AI-Powered Data Analytics Dashboard"
       width="75%" />
</p>

<p align="center">
  <strong>Production-style local analytics stack with AI-assisted querying & forecasting</strong>
</p>

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Relational_DB-blue?logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-black)
![Architecture](https://img.shields.io/badge/Architecture-End--to--End-success)
![Status](https://img.shields.io/badge/Status-Portfolio_Project-informational)

</p>

---

# 🧠 Executive Summary

**AI-Powered Data Analytics** adalah implementasi end-to-end analytics platform yang berjalan sepenuhnya secara lokal dan mengintegrasikan:

* Relational Database (PostgreSQL)
* Python analytics engine
* Local LLM (Ollama)
* AI-assisted SQL generation
* Forecasting (ARIMA)
* Interactive BI dashboard (Streamlit)

Project ini mensimulasikan arsitektur analytics modern yang biasanya ditemukan pada data platform skala production — namun dalam environment lokal.

> 🎯 Tujuan: Membangun AI-native analytics workflow tanpa ketergantungan cloud API eksternal.

---

# 🏗 High-Level Architecture

```
Data Sources (CSV / Excel)
            ↓
      PostgreSQL
            ↓
   Python Analytics Layer
            ↓
     Local LLM (Ollama)
            ↓
  AI Query + Insight Engine
            ↓
     Streamlit Dashboard
            ↓
     Business Decisions
```

---

# ✨ Key Capabilities

## 1️⃣ Natural Language → SQL Engine

* Mengubah pertanyaan natural language menjadi SQL
* Eksekusi langsung ke PostgreSQL
* Auto explanation query
* Context-aware prompting

Contoh:

> "Show top 5 products by revenue last month"

LLM akan:

* Generate optimized SQL
* Execute query
* Return structured result
* Explain logic behind query

---

## 2️⃣ AI-Assisted Data Analyst Mode

Tersedia 3 mode:

| Mode              | Function                         |
| ----------------- | -------------------------------- |
| General Assistant | Tanya jawab bebas                |
| Data Analyst      | Insight berbasis dataset         |
| SQL Expert        | Query optimization & explanation |

---

## 3️⃣ Forecasting Module (Time Series)

* ARIMA-based revenue forecasting
* Trend projection
* Demand estimation

Digunakan untuk simulasi predictive analytics use-case.

---

## 4️⃣ Interactive Dashboard

Fitur:

* Sidebar navigation
* Date range filtering
* AI temperature control
* Context injection toggle
* Multi-page analytics

---

# 📊 Technical Stack

| Layer       | Technology                |
| ----------- | ------------------------- |
| Backend     | Python                    |
| Database    | PostgreSQL                |
| AI Engine   | Ollama (mistral / llama2) |
| Dashboard   | Streamlit                 |
| Analytics   | Pandas, NumPy             |
| Forecasting | Statsmodels (ARIMA)       |
| Embedding   | nomic-embed-text          |

---

# 🔍 What Makes This Project Stand Out?

✅ 100% Local AI Stack
✅ No external API dependency
✅ Production-style layered architecture
✅ Clean separation of concerns
✅ Multi-mode AI interface
✅ Demonstrates Data + AI integration skill

Ini bukan sekadar dashboard — ini mini analytics platform.

---

# 📂 Project Structure

```
data-analyst-local/
│
├── dashboard/
│   └── streamlit_app.py
│
├── scripts/
│   ├── database_helper.py
│   ├── ollama_helper.py
│
├── notebooks/
├── data/
└── requirements.txt
```

---

# ⚙ Setup & Installation

```bash
git clone https://github.com/burhanudinera2018/data-analyst-local.git
cd data-analyst-local

pip install -r requirements.txt

# Pastikan PostgreSQL aktif
# Pastikan Ollama sudah pull model:
ollama pull mistral
ollama pull llama2

streamlit run dashboard/streamlit_app.py
```

---

# 📈 Business Impact Simulation

Dengan sistem ini, organisasi dapat:

* Mengurangi waktu query manual hingga 60%
* Memberikan self-service analytics
* Mempercepat insight generation
* Meningkatkan data accessibility untuk non-technical user

---

# 🧩 Future Improvements

* Role-based access control
* Vector database integration
* RAG pipeline enhancement
* Dockerized deployment
* CI/CD pipeline
* Cloud-ready version (GCP/AWS)

---

# 👨‍💻 Author

**Burhanudin Badiuzaman**
Data Analyst | AI Engineer (Aspirant) | Local LLM Enthusiast

🔗 [https://github.com/burhanudinera2018/data-analyst-local](https://github.com/burhanudinera2018/data-analyst-local)

---

# 📄 License

MIT License

---
