import streamlit as st
import joblib


# ========================================
# LOAD MODEL AND VECTORIZER
# ========================================

nb_model = joblib.load("news_classifier_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ========================================
# CATEGORY MAPPING
# ========================================

category_mapping = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}


# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="wide"
)


# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 40px;
}

.category-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.3);
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
    border: 1px solid rgba(128,128,128,0.3);
}

.result-title {
    font-size: 18px;
}

.result-category {
    font-size: 36px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ========================================
# HEADER
# ========================================

st.markdown(
    '<div class="main-title"> AI News Category Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An NLP-based machine learning system that automatically classifies '
    'news articles into four categories.'
    '</div>',
    unsafe_allow_html=True
)


# ========================================
# CATEGORY CARDS
# ========================================

st.subheader(" Available Categories")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        '<div class="category-card">'
        '<h3> World</h3>'
        '<p>International news</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="category-card">'
        '<h3> Sports</h3>'
        '<p>Sports and games</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="category-card">'
        '<h3> Business</h3>'
        '<p>Business and finance</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        '<div class="category-card">'
        '<h3> Sci/Tech</h3>'
        '<p>Science and technology</p>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ========================================
# NEWS INPUT
# ========================================

st.subheader(" Enter News Article")

news_text = st.text_area(
    "Paste your news article below:",
    height=220,
    placeholder=(
        "Example: Apple has announced a new artificial intelligence "
        "chip designed to improve machine learning performance..."
    )
)


# ========================================
# PREDICTION BUTTON
# ========================================

if st.button(" Predict News Category", use_container_width=True):

    if news_text.strip() == "":
        st.warning("Please enter a news article first.")

    else:

        # Convert news into TF-IDF
        news_tfidf = tfidf_vectorizer.transform([news_text])

        # Predict category
    prediction = nb_model.predict(news_tfidf)[0]

    # Get probabilities for all categories
    probabilities = nb_model.predict_proba(news_tfidf)[0]

    # Get confidence of predicted category
    confidence = max(probabilities) * 100

    # Convert number to category
    category = category_mapping[prediction]

    # Display result
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">
                 AI Prediction
            </div>
            <div class="result-category">
                {category}
            </div>
            <div class="result-title">
                Confidence: {confidence:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================
    # CATEGORY PROBABILITIES
    # ========================================

    st.subheader("📊 Category Probabilities")

    probability_data = {
        "World": probabilities[0],
        "Sports": probabilities[1],
        "Business": probabilities[2],
        "Sci/Tech": probabilities[3]
    }

    for category_name, probability in probability_data.items():
        st.write(f"**{category_name}** — {probability * 100:.2f}%")
        st.progress(float(probability))


# ========================================
# MODEL INFORMATION
# ========================================

st.divider()

st.subheader(" Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Model", "Naive Bayes")

with info2:
    st.metric("Accuracy", "90.07%")

with info3:
    st.metric("NLP Technique", "TF-IDF")


# ========================================
# FOOTER
# ========================================

st.divider()

st.caption(
    "AI News Category Classifier | Built using Python, NLP, "
    "Scikit-learn and Streamlit"
)

# ========================================
# HOW IT WORKS
# ========================================

st.divider()

st.subheader(" How This AI Classifier Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("""
    ### 1️. Input
    User enters a news article into the application.
    """)

with step2:
    st.markdown("""
    ### 2️. TF-IDF
    The text is converted into numerical features using TF-IDF.
    """)

with step3:
    st.markdown("""
    ### 3️. AI Model
    The trained Naive Bayes model analyzes the text features.
    """)

with step4:
    st.markdown("""
    ### 4️. Prediction
    The article is classified as World, Sports, Business, or Sci/Tech.
    """)

# ========================================
# PROJECT DETAILS
# ========================================

st.subheader(" Project Details")

st.markdown("""
**Dataset:** AG News Dataset

**Categories:** World, Sports, Business, Sci/Tech

**Feature Extraction:** TF-IDF

**Machine Learning Models:**
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

**Selected Prediction Model:** Multinomial Naive Bayes

**Test Accuracy:** 90.07%
""")