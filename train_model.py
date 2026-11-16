
import pandas as pd
import joblib

from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# 1. LOAD KAGGLE DATASET
# =====================================

print("Loading email dataset...")

try:
    data = pd.read_csv("email_text.csv")

    print("Dataset loaded successfully!")
    print("Total records:", len(data))

except FileNotFoundError:
    print("ERROR: email_text.csv not found!")
    print("Place the CSV file in your project folder.")
    raise

# Check required columns
required_columns = ["text", "label"]

if not all(column in data.columns for column in required_columns):
    raise ValueError(
        "Dataset must contain 'text' and 'label' columns."
    )

# =====================================
# 2. CLEAN DATASET
# =====================================

data = data.dropna(subset=["text", "label"]).copy()

data["text"] = data["text"].astype(str).str.strip()

data = data[data["text"] != ""]

data["label"] = pd.to_numeric(
    data["label"],
    errors="coerce"
)

data = data.dropna(subset=["label"])

data["label"] = data["label"].astype(int)

# Check valid labels
if not set(data["label"].unique()).issubset({0, 1}):
    raise ValueError("Labels must only contain 0 and 1.")

if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both labels: 0 and 1.")

print("\nDataset cleaned successfully!")
print("Remaining records:", len(data))
print("\nOriginal label distribution:")
print(data["label"].value_counts())

# =====================================
# 3. PREPARE DATA
# =====================================

X = data["text"]
y = data["label"]

# =====================================
# 4. SPLIT DATASET
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# 5. GENERATE 200 LEGITIMATE EMAILS
# =====================================

greetings = [
    "Hello John,",
    "Good morning,",
    "Hi Sarah,",
    "Dear Professor,",
    "Hello team,",
    "Good afternoon,",
    "Hi everyone,",
    "Dear colleagues,",
    "Hello Mary,",
    "Dear classmates,"
]

email_messages = [
    "I would like to schedule a meeting tomorrow to discuss our group project.",
    "Please review the attached document when you have time.",
    "I have completed my assigned task and uploaded the file.",
    "Can we arrange a short meeting to discuss the project requirements?",
    "Thank you for sending the information. I appreciate your help.",
    "Our next group discussion will take place on Friday afternoon.",
    "I would like to ask for clarification about the assignment instructions.",
    "Please let me know your available schedule for our consultation.",
    "I have updated the presentation and shared it with the group.",
    "Can you send me a copy of the notes from our previous meeting?",
    "I am writing to confirm my attendance at tomorrow's appointment.",
    "The final report is ready for review before submission.",
    "Please share your progress so we can prepare our presentation.",
    "I received your message and will respond after reviewing the details.",
    "Thank you for your feedback regarding our research paper.",
    "I would like to follow up on the documents submitted last week.",
    "The meeting has been moved to Monday at ten in the morning.",
    "Please check the revised schedule attached to this message.",
    "I will be available after class if you want to discuss the activity.",
    "Let me know if you have any questions about the project."
]

# Generate exactly 200 combinations
legitimate_emails = [
    f"{greeting} {message}"
    for greeting, message in product(
        greetings,
        email_messages
    )
]

# Create additional training data
additional_data = pd.DataFrame({
    "text": legitimate_emails,
    "label": [0] * len(legitimate_emails)
})

# Add examples only to training data
X_train = pd.concat(
    [X_train.reset_index(drop=True),
     additional_data["text"]],
    ignore_index=True
)

y_train = pd.concat(
    [y_train.reset_index(drop=True),
     additional_data["label"]],
    ignore_index=True
)

print("\nADDITIONAL TRAINING DATA")
print("--------------------------------")
print("Legitimate examples:", len(legitimate_emails))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# 6. CREATE ML MODEL
# =====================================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000,
        min_df=2
    )),
    ("classifier", MultinomialNB(alpha=0.1))
])

# =====================================
# 7. TRAIN MODEL
# =====================================

print("\nTraining Email Spam Detection Model...")

model.fit(X_train, y_train)

print("Training completed!")

# =====================================
# 8. TEST MODEL
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
))

# =====================================
# 9. TEST SAMPLE EMAILS
# =====================================

sample_emails = [
    "Hello John, I would like to schedule a meeting tomorrow to discuss our group project.",
    "Congratulations! You have won a free cash prize. Click here now to claim your reward!"
]

print("\nSAMPLE PREDICTIONS")
print("--------------------------------")

for email in sample_emails:

    result = model.predict([email])[0]
    probabilities = model.predict_proba([email])[0]

    label = "SPAM" if result == 1 else "NOT SPAM"

    confidence = max(probabilities) * 100

    print("\nEmail:", email)
    print("Prediction:", label)
    print("Confidence:", round(confidence, 2), "%")

# =====================================
# 10. SAVE TRAINED MODEL
# =====================================

joblib.dump(model, "model.pkl")

print("\n===================================")
print("MODEL SUCCESSFULLY SAVED!")
print("File: model.pkl")
print("===================================")
