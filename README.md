# 📰 Fake News Detection System

This project is a **Fake News Detection System** built using Python, Natural Language Processing (NLP), and multiple machine learning classifiers. It classifies news articles as either **real** or **fake**, helping to combat the spread of misinformation.

---

## 📁 Dataset Overview

The dataset is composed of two CSV files:

- **True.csv**: Contains articles from reliable news sources labeled as real.
- **Fake.csv**: Contains articles flagged or labeled as fabricated or deceptive.

Each file contains fields like `title`, `text`, `subject`, and `date`. The primary column used for classification is the `text` field, which holds the main body of the news article.

### Dataset Preparation Steps:
1. Load both datasets using `pandas`.
2. Add a new column `label`:
   - `0` for fake news
   - `1` for real news
3. Concatenate the datasets into one unified DataFrame.
4. Drop irrelevant columns like `title`, `subject`, and `date`.
5. Shuffle the dataset to mix fake and real articles.

---

## 🧹 Data Preprocessing

The data is cleaned using a custom text preprocessing function called `wordopt()`. It performs the following operations to prepare the data for machine learning:

### Cleaning Steps:
- Converts all characters to lowercase to maintain uniformity.
- Removes URLs and hyperlinks.
- Eliminates punctuation and special characters.
- Strips numbers to reduce noise.
- Removes extra spaces.

This step is crucial to standardize the text and improve the quality of feature extraction.

---

## 🔠 Feature Extraction using TF-IDF

To convert raw text into a numerical format understandable by ML models, **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization is used.

### Why TF-IDF?
- It assigns weights to words based on their frequency and importance across all documents.
- Helps reduce the influence of common words while boosting rarer, more meaningful terms.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
```

- The vectorized text is split into training and test sets using `train_test_split()`.

---

## 🤖 Machine Learning Models

Four machine learning classifiers were trained on the TF-IDF features to detect fake news:

### 1. Logistic Regression
- A linear model best suited for binary classification.
- Fast and interpretable.

### 2. Random Forest Classifier
- An ensemble of decision trees.
- Robust and handles overfitting better than single decision trees.

### 3. Decision Tree Classifier
- Splits data based on feature values.
- Easy to visualize and interpret but prone to overfitting.

### 4. Gradient Boosting Classifier
- Builds models sequentially to correct errors of previous models.
- High accuracy but computationally intensive.

---

## 📊 Results Comparison

Each model was evaluated on the test set using accuracy scores and classification metrics. Below is a summary comparison:

| Model                   | Accuracy (%) | Precision | Recall | F1-Score |
|------------------------|--------------|-----------|--------|----------|
| Logistic Regression    | 98.9         | 0.99      | 0.99   | 0.99     |
| Random Forest          | 99.6         | 1.00      | 0.99   | 0.99     |
| Decision Tree          | 99.7         | 1.00      | 1.00   | 1.00     |
| Gradient Boosting      | 99.8         | 1.00      | 1.00   | 1.00     |

### 🔍 Observations:
- **Gradient Boosting** slightly outperforms the others in accuracy but may be overfitting.
- **Logistic Regression** still performs very well and is fast to train.
- **Decision Tree** and **Random Forest** show near-perfect results, which may indicate the need for regularization or cross-validation.

---

## 🧪 Manual Testing Function

The system includes a `manual_testing()` function that allows users to manually input a news article and get predictions from all four models.

### Example:
```python
manual_testing("Breaking: Government announces new policy to reduce inflation.")
```

### Output:
Displays model-wise predictions (`Fake` or `Real`) for the input article.

---

## 📈 Recommended Improvements

To enhance the robustness and generalization of the models, the following techniques are suggested:

### ✅ Cross-Validation
Validate models on multiple subsets of data to ensure performance is consistent.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt, xv_train, y_train, cv=5)
print("Cross-validated accuracy:", scores.mean())
```

### ✅ Hyperparameter Tuning
Optimize model parameters using GridSearchCV for better performance.

```python
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(xv_train, y_train)
print(grid.best_params_)
```

### ✅ Improve Vectorization
Enhance the `TfidfVectorizer` by:
- Removing stop words: `stop_words='english'`
- Using n-grams for capturing adjacent word relationships: `ngram_range=(1,2)`

```python
vectorization = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
```

### ✅ Save Trained Models
Persist models using `joblib` to avoid retraining every time.

```python
import joblib
joblib.dump(lr, 'logistic_model.pkl')
```

---

## 📦 Installation & Requirements

Install required Python libraries with:

```bash
pip install pandas numpy scikit-learn
```

### Dependencies:
- `pandas`
- `numpy`
- `scikit-learn`
- `re` (regular expressions)

---

## 🧾 Conclusion

This Fake News Detection System successfully uses machine learning techniques to identify unreliable content in news articles. It demonstrates the power of NLP and classification algorithms in addressing misinformation. With a clean pipeline and scope for further tuning, it serves as a strong foundation for more advanced fake news detection projects.

---
