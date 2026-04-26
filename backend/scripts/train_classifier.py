import pandas as pd
import joblib
import logging
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    make_scorer
)

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "data/processed/cleaned_liar_dataset.csv"

logging.info("📥 Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["label"]  # Fake / Genuine

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Logistic Regression + Hyperparameter Tuning
# -----------------------------
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

base_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Explicit scorer with string labels
f1_scorer = make_scorer(f1_score, pos_label="Fake")

logging.info("🔍 Running GridSearchCV for Logistic Regression...")
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_tfidf, y_train)
model = grid_search.best_estimator_

logging.info(f"✅ Best C value found: {grid_search.best_params_['C']}")

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Fake")
recall = recall_score(y_test, y_pred, pos_label="Fake")
f1 = f1_score(y_test, y_pred, pos_label="Fake")
cm = confusion_matrix(y_test, y_pred)

# ROC-AUC requires numeric labels
y_test_binary = y_test.map(lambda x: 1 if x == "Fake" else 0)
y_pred_binary = pd.Series(y_pred).map(lambda x: 1 if x == "Fake" else 0)
auc = roc_auc_score(y_test_binary, y_pred_binary)

logging.info("📊 Model Evaluation Results")
logging.info(f"Accuracy  : {accuracy:.4f}")
logging.info(f"Precision : {precision:.4f}")
logging.info(f"Recall    : {recall:.4f}")
logging.info(f"F1 Score  : {f1:.4f}")
logging.info(f"ROC-AUC   : {auc:.4f}")
logging.info(f"Confusion Matrix:\n{cm}")

# -----------------------------
# Save Evaluation Report
# -----------------------------
REPORT_PATH = "results.txt"
with open(REPORT_PATH, "w") as f:
    f.write("Model Evaluation Report\n")
    f.write("=======================\n")
    f.write(f"Best C value : {grid_search.best_params_['C']}\n")
    f.write(f"Accuracy     : {accuracy:.4f}\n")
    f.write(f"Precision    : {precision:.4f}\n")
    f.write(f"Recall       : {recall:.4f}\n")
    f.write(f"F1 Score     : {f1:.4f}\n")
    f.write(f"ROC-AUC      : {auc:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

logging.info(f"📝 Evaluation report saved to {REPORT_PATH}")

# -----------------------------
# Save Model & Vectorizer
# -----------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, f"{MODEL_DIR}/fake_news_model.pkl")
joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

logging.info("✅ Model and vectorizer saved successfully")
