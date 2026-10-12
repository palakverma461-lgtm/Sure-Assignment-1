# ============================================================
# CUSTOMER SERVICE INTENT CLASSIFICATION
# Using Pandas + Python Built-in Libraries ONLY
# Algorithm: Naive Bayes (implemented from scratch)
# ============================================================

import pandas as pd
import re
import math
from collections import Counter


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_and_clean_data(
    train_path="Bitext_Sample_Customer_Service_Training_Dataset.csv",
    val_path="Bitext_Sample_Customer_Service_Validation_Dataset.csv",
    test_path="Bitext_Sample_Customer_Service_Testing_Dataset.csv"
):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    required_columns = {"utterance", "intent"}

    cleaned_dfs = []
    for name, df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df)
    ]:
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing columns: {missing_columns}"
            )

        df = df.dropna(subset=["utterance", "intent"]).copy()
        df["utterance"] = df["utterance"].astype(str).str.lower().str.strip()
        df["intent"] = df["intent"].astype(str).str.strip()
        df = df[(df["utterance"] != "") & (df["intent"] != "")].reset_index(drop=True)
        cleaned_dfs.append(df)

    return cleaned_dfs[0], cleaned_dfs[1], cleaned_dfs[2]


# ============================================================
# TOKENIZATION
# Split a sentence into individual words
# ============================================================

def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


# ============================================================
# NAIVE BAYES CLASSIFIER
# Learn word frequencies for each intent
# ============================================================

class NaiveBayesClassifier:

    def __init__(self):
        self.class_counts = Counter()
        self.word_counts = {}
        self.total_words = Counter()
        self.vocabulary = set()
        self.total_documents = 0
        self.classes = []
        self.vocab_size = 1
        self.intent_to_category = {}

    def fit(self, data):
        self.class_counts = Counter()
        self.word_counts = {}
        self.total_words = Counter()
        self.vocabulary = set()
        self.intent_to_category = {}

        has_category = "category" in data.columns

        # Fast tuple iteration for blazing fast training
        for row in data.itertuples(index=False):
            intent = getattr(row, "intent")
            utterance = getattr(row, "utterance")
            words = tokenize(utterance)

            # Count messages belonging to each intent
            self.class_counts[intent] += 1

            # Create a word counter for this intent
            if intent not in self.word_counts:
                self.word_counts[intent] = Counter()

            # Count words appearing in this intent
            self.word_counts[intent].update(words)

            # Count total words for this intent
            self.total_words[intent] += len(words)

            # Store all unique training words
            self.vocabulary.update(words)

            # Map intent to category if present
            if has_category:
                cat = getattr(row, "category", None)
                if pd.notna(cat) and intent not in self.intent_to_category:
                    self.intent_to_category[intent] = str(cat).strip()

        self.total_documents = sum(self.class_counts.values())
        self.classes = sorted(self.class_counts.keys())
        self.vocab_size = max(len(self.vocabulary), 1)

    def predict_scores(self, text):
        words = tokenize(text)
        scores = {}

        if not words:
            for intent in self.classes:
                scores[intent] = math.log(
                    self.class_counts[intent] / max(self.total_documents, 1)
                )
            return scores

        for intent in self.classes:
            score = math.log(
                self.class_counts[intent] / max(self.total_documents, 1)
            )
            denominator = self.total_words[intent] + self.vocab_size

            for word in words:
                word_count = self.word_counts[intent].get(word, 0)
                probability = (word_count + 1) / denominator
                score += math.log(probability)

            scores[intent] = score

        return scores

    def predict_proba(self, text):
        scores = self.predict_scores(text)
        if not scores:
            return {}
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        sum_exp = sum(exp_scores.values())
        return {k: (v / sum_exp) for k, v in exp_scores.items()}

    def get_top_intents(self, text, top_k=3):
        probas = self.predict_proba(text)
        sorted_intents = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        results = []
        for intent, prob in sorted_intents[:top_k]:
            cat = self.intent_to_category.get(intent, "GENERAL")
            results.append({
                "intent": intent,
                "probability": prob,
                "confidence_percent": round(prob * 100, 2),
                "category": cat
            })
        return results

    def predict_one(self, text):
        scores = self.predict_scores(text)
        if not scores:
            return None
        return max(scores, key=scores.get)

    def predict(self, texts):
        return [self.predict_one(text) for text in texts]


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate(data, dataset_name, model=None):
    if model is None:
        raise ValueError("Model must be provided for evaluation.")

    actual = data["intent"].tolist()
    predicted = model.predict(data["utterance"].tolist())
    total = len(actual)

    if total == 0:
        print(f"\n{dataset_name}: No samples to evaluate.")
        return 0.0

    correct = sum(a == p for a, p in zip(actual, predicted))
    accuracy = correct / total

    print("\n" + "=" * 75)
    print(f"{dataset_name} RESULTS")
    print("=" * 75)
    print("Total samples:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    labels = sorted(set(actual) | set(predicted))
    report_rows = []

    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        support = sum(a == label for a in actual)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        report_rows.append({
            "Intent": label,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Support": support
        })

    report_df = pd.DataFrame(report_rows)

    print("\nCLASSIFICATION REPORT")
    print("-" * 75)
    print(
        report_df.to_string(
            index=False,
            formatters={
                "Precision": "{:.2f}".format,
                "Recall": "{:.2f}".format,
                "F1-score": "{:.2f}".format
            }
        )
    )

    macro_precision = report_df["Precision"].mean()
    macro_recall = report_df["Recall"].mean()
    macro_f1 = report_df["F1-score"].mean()

    weighted_precision = (report_df["Precision"] * report_df["Support"]).sum() / total
    weighted_recall = (report_df["Recall"] * report_df["Support"]).sum() / total
    weighted_f1 = (report_df["F1-score"] * report_df["Support"]).sum() / total

    print("\nMACRO AVERAGE")
    print(
        "Precision:", round(macro_precision, 3),
        "| Recall:", round(macro_recall, 3),
        "| F1-score:", round(macro_f1, 3)
    )

    print("\nWEIGHTED AVERAGE")
    print(
        "Precision:", round(weighted_precision, 3),
        "| Recall:", round(weighted_recall, 3),
        "| F1-score:", round(weighted_f1, 3)
    )

    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for a, p in zip(actual, predicted):
        matrix.loc[a, p] += 1
    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"

    print("\nCONFUSION MATRIX")
    print(matrix.to_string())

    return accuracy


def load_or_train_model():
    train_df, val_df, test_df = load_and_clean_data()
    model = NaiveBayesClassifier()
    model.fit(train_df)
    return model, train_df, val_df, test_df


# ============================================================
# STANDALONE CLI SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":
    train_df, val_df, test_df = load_and_clean_data()
    print("=" * 60)
    print("DATASETS LOADED SUCCESSFULLY")
    print("=" * 60)
    print("Training dataset shape:", train_df.shape)
    print("Validation dataset shape:", val_df.shape)
    print("Testing dataset shape:", test_df.shape)
    print("\nColumn names:", train_df.columns.tolist())

    model = NaiveBayesClassifier()
    model.fit(train_df)
    print("\nMODEL TRAINING COMPLETED")
    print("Total training messages:", model.total_documents)
    print("Total intents:", len(model.classes))
    print("Vocabulary size:", model.vocab_size)

    evaluate(val_df, "VALIDATION", model=model)
    evaluate(test_df, "TEST", model=model)

    print("\n" + "=" * 60)
    print("CUSTOMER MESSAGE INFERENCE")
    print("=" * 60)
    print("Type 'exit' to stop.")

    while True:
        message = input("\nEnter a customer message: ").strip()
        if message.lower() == "exit":
            print("Inference stopped.")
            break
        if not message:
            print("Please enter a non-empty message.")
            continue
        predicted_intent = model.predict_one(message)
        print("Input message:", message)
        print("Predicted intent:", predicted_intent)
