# 📊 Streaming Sentiment Analysis App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://streaming-sentiment-analysis.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end data analytics and Natural Language Processing (NLP) project designed to scrape, preprocess, and analyze streaming data in real-time. This interactive dashboard provides businesses and analysts with immediate, actionable insights into public sentiment trends.

🔗 **Live Demo:** [streaming-sentiment-analysis.streamlit.app](https://streaming-sentiment-analysis.streamlit.app/)

---

## 🚀 Business Overview & Project Background
In today's fast-paced digital ecosystem, public opinion shifts rapidly. Monitoring brand reputation, feedback, or public response manually is no longer efficient. 

This project solves that challenge by building an automated pipeline that ingests data streams, applies advanced sentiment classification, and visualizes the results instantly. Analysts can leverage this tool to track campaign performance, detect public relations crises early, and drive data-backed strategic decisions.

---

## 🛠️ Tech Stack & Architecture

*   **Language & Core:** Python (Pandas, NumPy)
*   **Data Scraping & Extraction:** Automated ingestion pipelines
*   **Exploratory Data Analysis (EDA):** Matplotlib, Seaborn
*   **Natural Language Processing (NLP):** Scikit-learn, NLTK (Text preprocessing, Tokenization, Stopwords removal)
*   **Modeling:** Supervised Machine Learning (Sentiment Classification Model)
*   **Dashboard & Deployment:** Streamlit Community Cloud

---

## 📂 Repository Structure

```text
├── data/                  # Raw and processed datasets
├── data_preprocessing/    # Text cleaning, normalization, and tokenization scripts
├── data_scraping/         # Scripts for automated streaming and data gathering
├── eda/                   # Notebooks for Exploratory Data Analysis & insights
├── models/                # Saved model binaries and evaluation metrics
├── results/               # Output summaries and analysis reports
├── sentiment_modeling/    # Model training, tuning, and classification pipelines
├── topic_modeling/        # Unsupervised NLP for sub-theme discovery
├── visualization/         # Customized plot components for the dashboard
├── app.py                 # Main Streamlit application entry point
├── requirements.txt       # Project dependencies and libraries
└── README.md              # Project documentation
