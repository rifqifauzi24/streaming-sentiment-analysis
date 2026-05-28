import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Streaming Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv('data/reviews_preprocessed.csv')
    df = df.dropna(subset=['review_stemmed', 'sentiment'])
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load('../models/best_sentiment_model.pkl')
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    return model, vectorizer

df = load_data()
model, vectorizer = load_model()

# Color schemes
app_colors = {
    'Netflix': '#E50914',
    'Vidio': '#0075FF',
    'Disney+ Hotstar': '#113CCF',
    'WeTV': '#FF6600',
    'Viu': '#FCD000'
}

sentiment_colors = {
    'positive': '#2ECC71',
    'neutral': '#95A5A6',
    'negative': '#E74C3C'
}

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio(
    "Pilih halaman:",
    ["🏠 Home", "📊 EDA Explorer", "🤖 Sentiment Predictor", "🔍 Topic Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "Sentiment analysis of 5 streaming apps in Indonesia "
    "based on Google Play Store reviews."
)
st.sidebar.markdown("**Dataset:** 9,305 reviews")
st.sidebar.markdown("**Apps:** Netflix, Vidio, Disney+ Hotstar, WeTV, Viu")

# =========================
# PAGE 1: HOME
# =========================
if page == "🏠 Home":
    st.title("🎬 Streaming Wars Indonesia")
    st.subheader("Sentiment Analysis of OTT Platform Reviews")
    
    st.markdown("""
    Comparative sentiment analysis of user reviews from **5 major streaming platforms** 
    in Indonesia. This project covers end-to-end data science workflow: 
    web scraping → preprocessing → EDA → ML modeling → deployment.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("📱 Platforms", df['app_name'].nunique())
    with col3:
        avg_rating = df['rating'].mean()
        st.metric("⭐ Avg Rating", f"{avg_rating:.2f}/5")
    with col4:
        neg_pct = (df['sentiment'] == 'negative').mean() * 100
        st.metric("🔴 Negative %", f"{neg_pct:.1f}%")
    
    st.markdown("---")
    
    # Key findings
    st.subheader("🔑 Key Findings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("""
        **🚨 Disney+ Hotstar Crisis**
        - 86.7% negative sentiment
        - Lowest avg rating: 1.50/5
        - Drop drastis di Nov 2025
        """)
        
        st.warning("""
        **⚠️ Industry-wide Issues**
        - Semua platform rating < 3.0
        - Top complaints: iklan, login, pembayaran
        """)
    
    with col2:
        st.success("""
        **🏆 Vidio Best Performer**
        - 31.8% positive sentiment
        - Highest avg rating: 2.45/5
        - Local platform wins!
        """)
        
        st.info("""
        **🤖 Model Performance**
        - Best model: XGBoost
        - Accuracy: 80.3%
        - F1-Score: 0.765
        """)
    
    st.markdown("---")
    st.markdown("**👈 Explore each section in the sidebar to dive deeper!**")

# =========================
# PAGE 2: EDA EXPLORER
# =========================
elif page == "📊 EDA Explorer":
    st.title("📊 EDA Explorer")
    st.markdown("Explore the data interactively. Use filters on the sidebar.")
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Filters")
    selected_apps = st.sidebar.multiselect(
        "Pilih aplikasi:",
        options=df['app_name'].unique(),
        default=df['app_name'].unique()
    )
    
    selected_sentiment = st.sidebar.multiselect(
        "Pilih sentiment:",
        options=['positive', 'neutral', 'negative'],
        default=['positive', 'neutral', 'negative']
    )
    
    # Filter data
    df_filtered = df[
        df['app_name'].isin(selected_apps) &
        df['sentiment'].isin(selected_sentiment)
    ]
    
    st.info(f"📊 Showing {len(df_filtered):,} reviews based on filters")
    
    # Chart 1: Review volume per app
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Review Volume per App")
        app_counts = df_filtered['app_name'].value_counts()
        fig = px.bar(
            x=app_counts.index,
            y=app_counts.values,
            color=app_counts.index,
            color_discrete_map=app_colors,
            labels={'x': 'App', 'y': 'Number of Reviews'}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Average Rating per App")
        avg_rating = df_filtered.groupby('app_name')['rating'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=avg_rating.index,
            y=avg_rating.values,
            color=avg_rating.index,
            color_discrete_map=app_colors,
            labels={'x': 'App', 'y': 'Avg Rating'}
        )
        fig.update_layout(showlegend=False, height=400, yaxis_range=[0, 5])
        fig.add_hline(y=3, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 2: Sentiment proportion
    st.subheader("🎭 Sentiment Proportion per App")
    sentiment_pct = (df_filtered.groupby('app_name')['sentiment']
                     .value_counts(normalize=True).unstack() * 100)
    sentiment_pct = sentiment_pct.fillna(0)
    
    fig = go.Figure()
    for sent in ['positive', 'neutral', 'negative']:
        if sent in sentiment_pct.columns:
            fig.add_trace(go.Bar(
                name=sent,
                y=sentiment_pct.index,
                x=sentiment_pct[sent],
                orientation='h',
                marker_color=sentiment_colors[sent]
            ))
    fig.update_layout(
        barmode='stack',
        height=400,
        xaxis_title='Percentage (%)',
        yaxis_title=''
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Chart 3: Trend over time
    st.subheader("📈 Rating Trend Over Time")
    df_filtered['year_month'] = df_filtered['date'].dt.to_period('M').astype(str)
    monthly = df_filtered.groupby(['year_month', 'app_name'])['rating'].mean().reset_index()
    
    fig = px.line(
        monthly,
        x='year_month',
        y='rating',
        color='app_name',
        color_discrete_map=app_colors,
        markers=True
    )
    fig.update_layout(height=400, xaxis_title='', yaxis_title='Avg Rating')
    st.plotly_chart(fig, use_container_width=True)
    
    # Chart 4: Word cloud
    st.subheader("☁️ Word Cloud")
    sentiment_for_wc = st.selectbox("Pilih sentiment untuk word cloud:", 
                                      ['positive', 'negative', 'neutral'])
    
    text_wc = ' '.join(df_filtered[df_filtered['sentiment'] == sentiment_for_wc]
                       ['review_stemmed'].dropna().astype(str))
    
    if text_wc.strip():
        cmap = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Greys'}[sentiment_for_wc]
        wc = WordCloud(width=1200, height=400, background_color='white', 
                       colormap=cmap, max_words=100).generate(text_wc)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No data for selected filter combination.")

# =========================
# PAGE 3: SENTIMENT PREDICTOR
# =========================
elif page == "🤖 Sentiment Predictor":
    st.title("🤖 Sentiment Predictor")
    st.markdown("**Test the ML model:** Input a review and get sentiment prediction in real-time!")
    
    # Sample reviews for quick testing
    st.subheader("💡 Quick Test")
    sample = st.selectbox(
        "Try a sample review:",
        [
            "(Type your own below)",
            "aplikasi bagus sekali kualitas film mantap",
            "iklan terlalu banyak ganggu banget pas nonton",
            "tidak bisa login kode otp tidak masuk",
            "lumayan oke tapi langganan mahal",
            "film keren banyak pilihan suka banget",
            "buffering terus parah update aplikasi gagal"
        ]
    )
    
    # Input area
    if sample != "(Type your own below)":
        default_text = sample
    else:
        default_text = ""
    
    user_review = st.text_area(
        "✍️ Tulis review aplikasi streaming (Bahasa Indonesia):",
        value=default_text,
        height=120,
        placeholder="Contoh: aplikasinya bagus banget, banyak film menarik..."
    )
    
    if st.button("🚀 Predict Sentiment", type="primary"):
        if user_review.strip():
            # Transform input
            review_vec = vectorizer.transform([user_review.lower()])
            
            # Predict
            prediction = model.predict(review_vec)[0]
            
            # Get probability
            probas = model.predict_proba(review_vec)[0]
            classes = model.classes_
            
            # Display result
            st.markdown("---")
            
            # Emoji per sentiment
            emoji_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐'}
            color_map = {'positive': 'success', 'negative': 'error', 'neutral': 'warning'}
            
            # Handle XGBoost (numeric label)
            if isinstance(prediction, (int, np.integer)):
                try:
                    le = joblib.load('../models/label_encoder.pkl')
                    prediction = le.inverse_transform([prediction])[0]
                    classes = le.classes_
                except:
                    pass
            
            getattr(st, color_map.get(prediction, 'info'))(
                f"### {emoji_map.get(prediction, '🤖')} Prediction: **{prediction.upper()}**"
            )
            
            # Probability bar chart
            st.subheader("📊 Confidence Score")
            proba_df = pd.DataFrame({
                'Sentiment': classes,
                'Probability': probas
            }).sort_values('Probability', ascending=True)
            
            fig = px.bar(
                proba_df,
                x='Probability',
                y='Sentiment',
                orientation='h',
                color='Sentiment',
                color_discrete_map=sentiment_colors,
                text=proba_df['Probability'].apply(lambda x: f'{x*100:.1f}%')
            )
            fig.update_layout(showlegend=False, height=300, xaxis_range=[0, 1])
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please enter a review first!")
    
    st.markdown("---")
    st.markdown("**Model Info:**")
    st.markdown("- Algorithm: Best of (Logistic Regression / Random Forest / XGBoost)")
    st.markdown("- Trained on 9,305 Indonesian streaming app reviews")
    st.markdown("- Accuracy: 80.3% | F1-Score: 0.765")

# =========================
# PAGE 4: TOPIC INSIGHTS
# =========================
elif page == "🔍 Topic Insights":
    st.title("🔍 Topic Insights")
    st.markdown("Topic modeling results & actionable business insights.")
    
    tab1, tab2, tab3 = st.tabs(["🔴 Negative Topics", "🟢 Positive Topics", "💡 Recommendations"])
    
    with tab1:
        st.subheader("Top 5 Topics in NEGATIVE Reviews")
        
        topics_neg = {
            "💳 Pembayaran & Langganan": {
                "keywords": "langgan, bayar, vip, saldo, potong, dana",
                "insight": "User komplain soal sistem pembayaran, potongan saldo, dan biaya VIP yang tidak transparan."
            },
            "⭐ Kualitas Aplikasi Buruk": {
                "keywords": "aplikasi, jelek, buruk, sering",
                "insight": "Keluhan umum tentang kualitas aplikasi yang sering bermasalah."
            },
            "📺 Iklan Berlebihan": {
                "keywords": "iklan, tonton, banyak, menit",
                "insight": "Iklan terlalu sering dan panjang saat sedang menonton — keluhan #1."
            },
            "🎬 Konten Tidak Bisa Diakses": {
                "keywords": "padahal, buka, beli, paket, tiba",
                "insight": "User sudah bayar tapi tidak bisa akses konten — masalah serius."
            },
            "🔐 Login & OTP Issues": {
                "keywords": "login, masuk, susah, otp, daftar, kode, email",
                "insight": "Masalah authentication & registrasi — banyak yang gagal login."
            }
        }
        
        for topic, data in topics_neg.items():
            with st.expander(f"**{topic}**"):
                st.markdown(f"**Keywords:** `{data['keywords']}`")
                st.markdown(f"**Insight:** {data['insight']}")
    
    with tab2:
        st.subheader("Top 5 Topics in POSITIVE Reviews")
        
        topics_pos = {
            "🎬 Apresiasi Umum Film": {
                "keywords": "film, bagus, sangat, tonton, suka, good",
                "insight": "Pujian general tentang kualitas film dan aplikasi."
            },
            "📺 Drama Asia di Viu": {
                "keywords": "viu, drama, paket, suka",
                "insight": "User puas dengan drama Korea & Asia, terutama di Viu."
            },
            "✅ Aplikasi Lancar": {
                "keywords": "bagus, tonton, buka, lancar",
                "insight": "App responsive, buka cepat, nonton smooth."
            },
            "🌐 Subtitle Indonesia": {
                "keywords": "viu, subtitle, indo, langsung",
                "insight": "Apresiasi untuk subtitle Indo yang berkualitas — peluang differensiasi!"
            }
        }
        
        for topic, data in topics_pos.items():
            with st.expander(f"**{topic}**"):
                st.markdown(f"**Keywords:** `{data['keywords']}`")
                st.markdown(f"**Insight:** {data['insight']}")
    
    with tab3:
        st.subheader("💡 Strategic Recommendations")
        
        st.markdown("""
        ### 🎯 Untuk Platform Streaming Indonesia
        
        Berdasarkan analisis 9,305 review, ini adalah **5 pillar masalah** yang harus diatasi:
        """)
        
        recs = [
            ("💳 Transparansi Pembayaran", "Hilangkan hidden charges, jelaskan biaya VIP secara clear."),
            ("📺 Reduce Ad Frequency", "Iklan adalah keluhan #1. Pertimbangkan ad-free tier murah."),
            ("🎬 Content Accessibility", "Konten yang sudah dibayar harus 100% accessible — bug ini fatal."),
            ("🔐 Reliable Authentication", "Sistem OTP & login harus stable, terutama saat traffic tinggi."),
            ("🌐 Quality Subtitle", "Indonesian subtitle is a differentiator — invest in quality.")
        ]
        
        for title, desc in recs:
            st.markdown(f"**{title}**")
            st.markdown(f"_{desc}_")
            st.markdown("")
        
        st.markdown("---")
        st.info("""
        🚨 **Priority Action:** Disney+ Hotstar perlu investigasi mendalam — 
        86.7% sentiment negatif dan rating drop drastis di Nov 2025 menunjukkan ada masalah serius 
        yang perlu di-address segera.
        """)

# =========================
# FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:** Python, scikit-learn, Streamlit")
st.sidebar.markdown("[View on GitHub](https://github.com/yourusername/streaming-sentiment-analysis)")