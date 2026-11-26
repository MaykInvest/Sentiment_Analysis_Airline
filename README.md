# ✈️ U.S. Airline Tweet Sentiment Analysis

Classifying airline-related tweets as **negative**, **neutral** or **positive**, and comparing classical machine learning (TF-IDF + Logistic Regression) against transformers (a ready-made tweet model and a fine-tuned DistilBERT).

> **Main result:** fine-tuned DistilBERT reached **83.9% accuracy / 0.789 macro F1** on a held-out test set, about 6 points ahead of the best classical model (77.2% / 0.726).

---

## 📌 Project Goal

Customers complain, praise and ask questions about airlines on social media. This project asks:

1. How well can a classical, interpretable model classify tweet sentiment?
2. How much does a transformer improve on it?
3. Does a ready-made, pretrained tweet model (no training on this data) beat a model fine-tuned on in-domain data?

## 📂 Data

- **Source:** *Twitter US Airline Sentiment* dataset (Kaggle).
- **Size after cleaning:** 14,427 tweets (177 duplicate texts removed).
- **Classes:** negative (~63%), neutral, positive, so the data is clearly imbalanced.
- **Split:** 80% train / 20% test, stratified (train 11,541, test 2,886). For DistilBERT, 10% of train was held out as a validation set for early stopping.

## 🔬 Methodology

| Step | What was done |
|---|---|
| Cleaning | Lowercasing, removing @mentions, URLs and punctuation, tokenization, stopword removal and lemmatization. **Negation words ("no", "not", ...) are kept**, since they carry sentiment. |
| Duplicates | Removed duplicate tweet texts **before** splitting to avoid train/test leakage. |
| Features | TF-IDF (unigrams + bigrams) plus airline, retweet count and tweet length, inside a scikit-learn `Pipeline` / `ColumnTransformer`. |
| Baseline | `DummyClassifier`. |
| Classical models | Logistic Regression, LinearSVC, Naive Bayes, compared with stratified 5-fold cross-validation. |
| Tuning | `GridSearchCV` on Logistic Regression (`C`, `class_weight`), scored by **macro F1**. |
| Transformers | (a) `cardiffnlp/twitter-roberta-base-sentiment-latest`, used as is with no training. (b) `distilbert-base-uncased` fine-tuned with Hugging Face `Trainer` (4 epochs max, early stopping, best epoch by macro F1) on a Colab T4 GPU. |
| Evaluation | All final models scored on the **same 2,886 test tweets, used once**. |

Random Forest and XGBoost were also tried in an earlier run. Random Forest trailed the linear models (accuracy 0.754) and XGBoost roughly matched Logistic Regression (0.773) but trained much slower, so both were dropped from the final comparison.

## 📊 Results (test set)

| Model | Accuracy | F1 weighted | F1 macro |
|---|---|---|---|
| Logistic Regression (TF-IDF, tuned) | 0.7717 | 0.7783 | 0.7255 |
| RoBERTa for tweets, ready-made (no training) | 0.7651 | 0.7758 | 0.7255 |
| **DistilBERT (fine-tuned)** | **0.8385** | **0.8368** | **0.7891** |

**DistilBERT per-class (test):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.89 | 0.91 | 0.90 |
| neutral | 0.71 | 0.65 | 0.68 |
| positive | 0.77 | 0.80 | 0.79 |

### Key takeaways

- **Fine-tuning on in-domain data gave the best results.** DistilBERT beat the tuned classical model by about 6 points in accuracy and macro F1. With 2,886 test tweets, this gap is well outside random noise.
- **A ready-made tweet model with no training only matched the tuned Logistic Regression**, so training on this dataset's labels mattered more than general pretraining on tweets.
- **Neutral is the hardest class** (F1 0.68), since neutral tweets are often informational and sit between the other two classes.
- Fine-tuning took about 2 minutes on a T4 GPU.

## ⚠️ Limitations

- Results come from a **single training run and a single data split**, so read the gap as "about 6 points", not an exact figure.
- Macro F1 is the fairer metric here because of class imbalance.
- The demo predictions on a handful of hand-written tweets are illustrations, not evidence. **Sarcasm handling was not properly tested.**
- The ready-made model was pretrained on a large amount of Twitter data, so it is "pretrained, no fine-tuning", not a like-for-like comparison with models trained here.

## 🚀 How to Run

1. Open the notebook in Google Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. Run **Runtime → Restart session**, then **Run all**.
3. The notebook downloads the dataset, trains the classical models and fine-tunes DistilBERT (about 2 minutes on a T4).

> Replace the filename in the badge below with your notebook's name.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaykInvest/Sentiment_Analysis_Airline/blob/main/YOUR_NOTEBOOK.ipynb)

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · NLTK · PyTorch · Hugging Face Transformers & Datasets · Matplotlib · Seaborn

## 🔮 Possible Next Steps

- Run several random seeds to measure variance.
- Fine-tune a larger model (for example RoBERTa) and compare.
- Inspect DistilBERT's neutral-class errors and test sarcasm explicitly.
- Wrap the model in a small demo app.

## 🤝 Acknowledgements

This project was built with assistance from Claude (Anthropic) for guidance and code review.
