AI News Classifier

A machine learning project that classifies news articles into four categories:

- Business
- World
- Sports
- Sci/Tech

The project uses Natural Language Processing (NLP), TF-IDF vectorization, and Machine Learning for text classification.

Technologies

- Python
- Scikit-learn
- Streamlit
- Joblib
- NumPy

Dataset

AG News dataset containing 7,600 news articles with 1,900 articles from each category.

Features used:
- Title
- Description

Machine Learning Models

| Model | Accuracy |
|---|---:|
| Logistic Regression | 90.00% |
| Multinomial Naive Bayes | 90.07% |
| Linear SVM | 89.87% |

Multinomial Naive Bayes was selected as the final model.

Project Structure

AI_NEWS_CLASSIFIER/
│
├── AG news dataset.csv
├── app.py
├── main.py
├── predict.py
├── news_classifier_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md

Run Locally

Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app.py


Live Application Link

https://ainewsclassifier-swnfhp8mna6i6wzq3ezayc.streamlit.app/
