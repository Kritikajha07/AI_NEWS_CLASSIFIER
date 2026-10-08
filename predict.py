import joblib

# Load the saved model and TF-IDF vectorizer
nb_model = joblib.load("news_classifier_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")


category_mapping = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}


def predict_news(news_text):
    # Convert the input news into TF-IDF features
    news_tfidf = tfidf_vectorizer.transform([news_text])

    # Predict the category
    prediction = nb_model.predict(news_tfidf)[0]

    # Convert category number into category name
    category = category_mapping[prediction]

    print("\n========================================")
    print("AI NEWS CATEGORY CLASSIFIER")
    print("========================================")
    print("News:", news_text)
    print("Predicted Category:", category)


# Take news input from the user
new_news = input("Enter a news article: ")

predict_news(new_news)