import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Download required NLTK resources
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Label Mapping (Option A)
# -----------------------------
FAKE_LABELS = {"pants-fire", "false", "barely-true", "half-true"}
GENUINE_LABELS = {"mostly-true", "true"}

def map_label(label: str) -> str | None:
    if label in FAKE_LABELS:
        return "Fake"
    if label in GENUINE_LABELS:
        return "Genuine"
    return None

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(tokens)

# -----------------------------
# Load & Process Dataset
# -----------------------------
def load_and_process(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=None)

    if df.shape[1] < 3:
        raise ValueError(
            f"Unexpected dataset format in {file_path}. "
            "Expected at least 3 columns."
        )

    df = df[[1, 2]]
    df.columns = ["label", "text"]

    df["label"] = df["label"].apply(map_label)
    df.dropna(inplace=True)

    df["text"] = df["text"].apply(clean_text)

    logging.info(f"Processed {len(df)} rows from {file_path}")
    return df

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    logging.info("Starting preprocessing of LIAR dataset")

    train_df = load_and_process("data_raw/train.tsv")
    valid_df = load_and_process("data_raw/valid.tsv")
    test_df = load_and_process("data_raw/test.tsv")

    full_df = pd.concat([train_df, valid_df, test_df], axis=0)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = "data/processed/cleaned_liar_dataset.csv"
    full_df.to_csv(output_path, index=False)

    logging.info("Preprocessing complete")
    logging.info(f"Saved cleaned dataset to: {output_path}")
    logging.info(f"Total samples: {len(full_df)}")
