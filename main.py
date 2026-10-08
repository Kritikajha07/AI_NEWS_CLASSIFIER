import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("AG news dataset.csv")

print("Dataset loaded successfully!")

# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 4. CATEGORY MAPPING
# ==========================================

category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

df["Category"] = df["Class Index"].map(category_names)

# ==========================================
# 5. CATEGORY DISTRIBUTION
# ==========================================

print("\nCategory Distribution:")
print(df["Category"].value_counts())

# ==========================================
# 6. CHECK DUPLICATES
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# 7. DISPLAY EXAMPLES
# ==========================================

print("\nExamples from each category:")

for category in category_names.values():

    sample = df[df["Category"] == category].iloc[0]

    print("\n==============================")
    print("CATEGORY:", category)
    print("TITLE:", sample["Title"])
    print("DESCRIPTION:", sample["Description"][:250])

# ==========================================
# 8. VISUALIZE CATEGORY DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Category",
    order=["World", "Sports", "Business", "Sci/Tech"]
)

plt.title("AG News Category Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")


# ==========================================
# 9. CREATE TEXT COLUMN FOR NLP
# ==========================================

df["text"] = (
    df["Title"].fillna("") + " " +
    df["Description"].fillna("")
)

print("\nCombined Text Example:")
print(df["text"].iloc[0])

print("\nCategory and Text:")
print(df[["Category", "text"]].head())

# ==========================================
# 10. SEPARATE INPUT AND OUTPUT
# ==========================================

X = df["text"]
y = df["Class Index"]

print("\nInput (X) example:")
print(X.iloc[0])

print("\nOutput (y) example:")
print(y.iloc[0])

# ==========================================
# 10. TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X = df["text"]
y = df["Class Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 11. TF-IDF VECTORIZATION
# ==========================================

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ==========================================
# 12. TRAIN AI MODEL
# ==========================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# ==========================================
# 13. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)

print("\nFirst 20 Predictions:")
print(y_pred[:20])


# ==========================================
# 14. EVALUATE MODEL
# ==========================================

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "World",
            "Sports",
            "Business",
            "Sci/Tech"
        ]
    )
)

# ==========================================
# 16. CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["World", "Sports", "Business", "Sci/Tech"],
    yticklabels=["World", "Sports", "Business", "Sci/Tech"]
)

plt.title("Confusion Matrix - AG News Classification")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.show()
# ==========================================
# 17. NAIVE BAYES MODEL
# ==========================================

from sklearn.naive_bayes import MultinomialNB

# Create Naive Bayes model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

print("\nNaive Bayes model training completed!")

# Make predictions
nb_pred = nb_model.predict(X_test_tfidf)

# Evaluate
nb_accuracy = accuracy_score(y_test, nb_pred)

print("\n========================================")
print("NAIVE BAYES PERFORMANCE")
print("========================================")

print("Accuracy:", nb_accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        nb_pred,
        target_names=[
            "World",
            "Sports",
            "Business",
            "Sci/Tech"
        ]
    )
)
# ==========================================
# 18. LINEAR SVM MODEL
# ==========================================

from sklearn.svm import LinearSVC

# Create Linear SVM model
svm_model = LinearSVC()

# Train the model
svm_model.fit(X_train_tfidf, y_train)

print("\nLinear SVM model training completed!")

# Make predictions
svm_pred = svm_model.predict(X_test_tfidf)

# Evaluate
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n========================================")
print("LINEAR SVM PERFORMANCE")
print("========================================")

print("Accuracy:", svm_accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        svm_pred,
        target_names=[
            "World",
            "Sports",
            "Business",
            "Sci/Tech"
        ]
    )
)
# ========================================
# MODEL COMPARISON
# ========================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Naive Bayes Accuracy:         {nb_accuracy:.4f}")
print(f"Linear SVM Accuracy:          {svm_accuracy:.4f}")

# ========================================
# PREDICT NEW NEWS
# ========================================

# ========================================
# PREDICT NEW NEWS
# ========================================

category_mapping = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
def predict_news(news_text):
    
    # Convert new news text into TF-IDF features
    news_tfidf = tfidf_vectorizer.transform([news_text])

    # Predict category using Naive Bayes
    prediction = nb_model.predict(news_tfidf)[0]

    # Convert category number into category name
    category = category_mapping[prediction]

    print("\n========================================")
    print("NEWS CLASSIFICATION")
    print("========================================")
    print("News:", news_text)
    print("Predicted Category:", category)


# ========================================
# INTERACTIVE NEWS CLASSIFIER
# ========================================

print("\n========================================")
print("AI NEWS CATEGORY CLASSIFIER")
print("========================================")

new_news = input("Enter a news article: ")

predict_news(new_news)

# ========================================
# SAVE MODEL AND VECTORIZER
# ========================================

joblib.dump(nb_model, "news_classifier_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and TF-IDF vectorizer saved successfully!")

plt.show()