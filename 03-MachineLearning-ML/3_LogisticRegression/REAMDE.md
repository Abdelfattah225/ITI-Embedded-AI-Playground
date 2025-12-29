
# Logistic Regression - Complete Guide

## 📌 What is Logistic Regression?

Logistic Regression predicts **categorical outcomes** (classification) using probability.

### Formula

Step 1: z = w₁x₁ + w₂x₂ + ... + b (same as linear)
Step 2: probability = 1 / (1 + e^(-z)) (sigmoid function)
Step 3: if probability > 0.5 → Class 1, else → Class 0

text


---

## 🎯 When to Use Logistic Regression?

| Use Logistic Regression | DON'T Use |
|-------------------------|-----------|
| Spam/Not Spam | House price prediction |
| Disease/Healthy | Salary prediction |
| Survived/Died | Temperature forecast |
| Pass/Fail | Any continuous number |

**Rule:** Target variable must be **categorical** (usually binary: 0/1)

---

## 📊 Complete Pipeline

    Load Data
    EDA (Exploratory Data Analysis)
    Encode Categorical Variables
    Split Data (train/test)
    Scale Numerical Features
    Train Model
    Evaluate (Accuracy, Confusion Matrix, Classification Report)

text


⚠️ **Important:** Split BEFORE Scaling to avoid data leakage!

---

## 📈 Evaluation Metrics

### Confusion Matrix

text

             PREDICTED
            No    Yes
      ┌────────┬────────┐
 No   │   TN   │   FP   │

ACTUAL ├────────┼────────┤
Yes │ FN │ TP │
└────────┴────────┘

TN = True Negative (Correctly predicted No)
TP = True Positive (Correctly predicted Yes)
FP = False Positive (Predicted Yes, was No) - Type I Error
FN = False Negative (Predicted No, was Yes) - Type II Error

text


### Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP + TN) / Total | Overall correctness |
| Precision | TP / (TP + FP) | "Of predicted YES, how many correct?" |
| Recall | TP / (TP + FN) | "Of actual YES, how many did we find?" |
| F1-Score | 2 × (P × R) / (P + R) | Balance of Precision & Recall |

---

## 🎯 When to Prioritize Which Metric?

### High RECALL When:
- Missing positive cases is dangerous
- Examples: Cancer detection, Fraud detection, Security threats
- "Better safe than sorry - catch all positives!"

### High PRECISION When:
- False alarms are costly
- Examples: Spam filter, Recommendations
- "Only flag when we're sure!"

---

## 🔄 Overfitting vs Underfitting

┌─────────────────────────────────────────────────────────────┐
│ │
│ GOOD FIT: Train: 90% Test: 88% (small gap) │
│ OVERFITTING: Train: 98% Test: 65% (large gap) │
│ UNDERFITTING: Train: 55% Test: 52% (both low) │
│ │
└─────────────────────────────────────────────────────────────┘

text


---

## 💻 Complete Code Template

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("data.csv")

# 2. EDA
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()

# 3. Handle Missing Values
df['num_col'].fillna(df['num_col'].mean(), inplace=True)
df['cat_col'].fillna(df['cat_col'].mode()[0], inplace=True)

# 4. Remove Duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# 5. Encode Categorical Variables
# Binary
df['binary_col'] = df['binary_col'].replace({'yes': 1, 'no': 0})

# One-Hot (for multiple categories with no order)
df = pd.get_dummies(df, columns=['category_col'])

# 6. Separate Features and Target
x = df.drop('target', axis=1)
y = df['target']

# 7. Split Data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=42
)

# 8. Scale Numerical Features Only
scaler = StandardScaler()
numerical_cols = ['col1', 'col2']

x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])

# 9. Train Model
model = LogisticRegression()
model.fit(x_train, y_train)

# 10. Predict
y_pred = model.predict(x_test)

# 11. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

❓ Common Interview Questions
Q1: What is the difference between Linear and Logistic Regression?

A: Linear predicts continuous numbers (price, salary). Logistic predicts categories (yes/no, spam/not spam) using sigmoid function.
Q2: What is the Sigmoid function?

A: σ(z) = 1/(1+e^(-z)). It converts any number to a probability between 0 and 1.
Q3: What is a Confusion Matrix?

A: A table showing TP, TN, FP, FN - comparing actual vs predicted values.
Q4: What is the difference between Precision and Recall?

A: Precision = Of predicted positives, how many correct. Recall = Of actual positives, how many we found.
Q5: When is Recall more important than Precision?

A: When missing a positive case is dangerous (medical diagnosis, fraud detection).
Q6: What is overfitting?

A: Model memorizes training data instead of learning patterns. High training accuracy, low test accuracy.
Q7: What is F1-Score?

A: Harmonic mean of Precision and Recall. Balances both metrics.
🎯 Quick Decision Checklist

Is target variable categorical? → Use Logistic Regression
Are there categorical columns? → Encode them
Did I split BEFORE scaling? → Avoid data leakage
Did I use fit_transform on train only? → Prevent leakage
Did I check confusion matrix? → Understand errors

    Did I print classification report? → See Precision/Recall

📊 Metrics Comparison
Scenario	Focus On
Medical diagnosis	High Recall
Spam filter	High Precision
Balanced importance	F1-Score
Overall performance	Accuracy