# Logistic Regression - Complete Guide 📚

## 📌 What is Logistic Regression?

Logistic Regression predicts **categorical outcomes** (classification) using probability.

### Formula

```
Step 1: z = w₁x₁ + w₂x₂ + ... + b (same as linear regression)
Step 2: probability = 1 / (1 + e^(-z)) (sigmoid function)
Step 3: if probability > 0.5 → Class 1, else → Class 0
```

### Sigmoid Function Visualization

```
Probability
    │
1.0 │                    _______________
    │                  /
0.5 │─────────────────/──────────────────
    │               /
0.0 │______________/
    │
    └─────────────────────────────────────── z
         -∞            0            +∞

Output is ALWAYS between 0 and 1!
```

---

## 🎯 When to Use Logistic Regression?

| ✅ Use Logistic Regression | ❌ DON'T Use |
|---------------------------|-------------|
| Spam / Not Spam | House price prediction |
| Disease / Healthy | Salary prediction |
| Survived / Died | Temperature forecast |
| Pass / Fail | Any continuous number |
| Fraud / Not Fraud | Stock price prediction |

**Rule:** Target variable must be **categorical** (binary: 0/1 or multiclass: 0/1/2/...)

---

## 📊 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load Data                                               │
│        ↓                                                    │
│  2. EDA (Exploratory Data Analysis)                        │
│        ↓                                                    │
│  3. Handle Missing Values                                   │
│        ↓                                                    │
│  4. Encode Categorical Variables                            │
│        ↓                                                    │
│  5. Split Data (train/test)  ← BEFORE scaling!             │
│        ↓                                                    │
│  6. Scale Numerical Features                                │
│        ↓                                                    │
│  7. Train Model                                             │
│        ↓                                                    │
│  8. Predict                                                 │
│        ↓                                                    │
│  9. Evaluate                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

⚠️ **Important:** Split BEFORE Scaling to avoid data leakage!

---

## 🔤 Encoding Types

| Type | When to Use | Example |
|------|-------------|---------|
| Binary | Exactly 2 categories | Yes/No → 1/0 |
| One-Hot | Multiple categories, NO order | Colors → [1,0,0], [0,1,0], [0,0,1] |
| Ordinal | Multiple categories WITH order | Low=1, Medium=2, High=3 |

---

## 📈 Evaluation Metrics

### Confusion Matrix

```
                     PREDICTED
                    No      Yes
              ┌─────────┬─────────┐
         No   │   TN    │   FP    │
  ACTUAL      ├─────────┼─────────┤
         Yes  │   FN    │   TP    │
              └─────────┴─────────┘

TN = True Negative  → Correctly predicted No  ✓
TP = True Positive  → Correctly predicted Yes ✓
FP = False Positive → Predicted Yes, was No   ✗ (Type I Error)
FN = False Negative → Predicted No, was Yes   ✗ (Type II Error)
```

### Key Metrics Formulas

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP + TN) / Total | Overall correctness |
| Precision | TP / (TP + FP) | "Of predicted YES, how many correct?" |
| Recall | TP / (TP + FN) | "Of actual YES, how many did we find?" |
| F1-Score | 2 × (P × R) / (P + R) | Balance of Precision & Recall |

### How to Remember Precision vs Recall

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  PRECISION:                                                 │
│  "Of all I PREDICTED positive, how many are truly positive?"│
│  Focus: Quality of my predictions                           │
│  Enemy: False Positives (false alarms)                      │
│                                                             │
│  RECALL:                                                    │
│  "Of all ACTUAL positives, how many did I catch?"          │
│  Focus: Finding all positive cases                          │
│  Enemy: False Negatives (missed cases)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 When to Prioritize Which Metric?

### High RECALL When:
- Missing positive cases is **dangerous**
- Examples:
  - 🏥 Cancer detection → Don't miss sick patients!
  - 💳 Fraud detection → Don't miss any fraud!
  - ✈️ Airport security → Don't miss threats!
- Motto: *"Better safe than sorry - catch all positives!"*

### High PRECISION When:
- False alarms are **costly**
- Examples:
  - 📧 Spam filter → Don't block important emails!
  - 🎬 Recommendations → Don't annoy users with bad suggestions!
- Motto: *"Only flag when we're sure!"*

### Quick Reference Table

| Scenario | Prioritize | Why |
|----------|------------|-----|
| Medical diagnosis | High Recall | Missing sick patient is dangerous |
| Spam filter | High Precision | Blocking good email is costly |
| Fraud detection | High Recall | Missing fraud causes money loss |
| Product recommendations | High Precision | Bad recommendations annoy users |
| Balanced importance | F1-Score | Balance both metrics |

---

## 🔄 Overfitting vs Underfitting vs Good Fit

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  UNDERFITTING          GOOD FIT           OVERFITTING       │
│                                                             │
│      ●    ●           ●    ●               ●    ●          │
│    ●        ●       ●        ●           ●  ╲╱    ●        │
│  ──────────────     ╱──────────╲         ╱ ╲  ╱╲  ╲       │
│    ●    ●          ●            ●       ●   ╲╱  ╲╱  ●      │
│      ●                ●      ●                  ●          │
│                                                             │
│  Too simple!        Just right!         Too complex!       │
│  Can't learn        Learns pattern      Memorizes noise    │
│                                                             │
│  Train: 55%         Train: 90%          Train: 98%         │
│  Test:  52%         Test:  88%          Test:  65%         │
│  Gap:   3%          Gap:   2%           Gap:   33%         │
│  (both low)         (both high)         (large gap)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📏 Scaling Rules

```
┌─────────────────────────────────────────────────────────────┐
│              WHAT TO SCALE                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ SCALE:                                                  │
│     • Continuous numerical with large range                │
│     • Examples: age, salary, price, area, fare             │
│                                                             │
│  ❌ DON'T SCALE:                                            │
│     • Binary columns (0/1)                                 │
│     • One-hot encoded columns (0/1)                        │
│     • Columns with very small range (1, 2, 3)              │
│                                                             │
│  ⚠️ REMEMBER:                                               │
│     • fit_transform() on training data                     │
│     • transform() only on test data                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

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

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("data.csv")

# ============================================================
# 2. EDA (EXPLORATORY DATA ANALYSIS)
# ============================================================
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nTarget Distribution:")
print(df['target'].value_counts())

# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================
# For numerical columns: use mean
df['num_col'].fillna(df['num_col'].mean(), inplace=True)

# For categorical columns: use mode
df['cat_col'].fillna(df['cat_col'].mode()[0], inplace=True)

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================
# Binary encoding (2 categories)
df['binary_col'] = df['binary_col'].replace({'yes': 1, 'no': 0})

# One-Hot encoding (multiple categories, no order)
df = pd.get_dummies(df, columns=['category_col'])

# Ordinal encoding (multiple categories with order)
df['size'] = df['size'].replace({'small': 1, 'medium': 2, 'large': 3})

# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================
x = df.drop('target', axis=1)
y = df['target']

# ============================================================
# 7. SPLIT DATA (BEFORE SCALING!)
# ============================================================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=42
)

# ============================================================
# 8. SCALE NUMERICAL FEATURES ONLY
# ============================================================
scaler = StandardScaler()
numerical_cols = ['col1', 'col2', 'col3']

x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])

# ============================================================
# 9. TRAIN MODEL
# ============================================================
model = LogisticRegression()
model.fit(x_train, y_train)

# ============================================================
# 10. PREDICT
# ============================================================
y_pred = model.predict(x_test)

# ============================================================
# 11. EVALUATE
# ============================================================
print("="*50)
print("MODEL EVALUATION")
print("="*50)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 12. VISUALIZE CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## 🔀 Binary vs Multiclass Classification

```
┌─────────────────────────────────────────────────────────────┐
│         BINARY vs MULTICLASS                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BINARY (2 classes):                                       │
│  • Target: 0 or 1                                          │
│  • Confusion Matrix: 2×2                                   │
│  • Example: Spam/Not Spam, Disease/Healthy                 │
│                                                             │
│  MULTICLASS (3+ classes):                                  │
│  • Target: 0, 1, 2, ... (multiple classes)                │
│  • Confusion Matrix: N×N                                   │
│  • Example: Iris species (Setosa/Versicolor/Virginica)    │
│                                                             │
│  Same code! sklearn handles it automatically!              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ❓ Common Interview Questions

### Q1: What is the difference between Linear and Logistic Regression?
**A:** Linear Regression predicts continuous numbers (price, salary). Logistic Regression predicts categories (yes/no, spam/not spam) using the sigmoid function to output probabilities between 0 and 1.

### Q2: What is the Sigmoid function?
**A:** σ(z) = 1/(1+e^(-z)). It converts any number to a probability between 0 and 1. This allows us to interpret the output as a probability of belonging to class 1.

### Q3: What is a Confusion Matrix?
**A:** A table showing TP (True Positive), TN (True Negative), FP (False Positive), and FN (False Negative) - comparing actual vs predicted values to understand where the model makes mistakes.

### Q4: What is the difference between Precision and Recall?
**A:** 
- Precision = Of all predicted positives, how many are actually positive? (TP / (TP + FP))
- Recall = Of all actual positives, how many did we find? (TP / (TP + FN))

### Q5: When is Recall more important than Precision?
**A:** When missing a positive case is dangerous or costly. Examples: medical diagnosis (don't miss sick patients), fraud detection (don't miss fraud), security threats (don't miss threats).

### Q6: When is Precision more important than Recall?
**A:** When false alarms are costly or annoying. Examples: spam filter (don't block important emails), product recommendations (don't suggest bad products).

### Q7: What is overfitting?
**A:** When a model memorizes training data instead of learning patterns. Signs: High training accuracy (98%) but low test accuracy (65%). The model fails to generalize to new data.

### Q8: What is F1-Score?
**A:** The harmonic mean of Precision and Recall: F1 = 2 × (P × R) / (P + R). It balances both metrics and is useful when you need a single score to compare models.

### Q9: What is data leakage?
**A:** When information from test data accidentally influences the training process. Example: scaling all data before splitting. This leads to fake high accuracy that won't work in production.

### Q10: Why do we use fit_transform on training data but only transform on test data?
**A:** fit_transform learns parameters (mean, std) from training data and applies them. We use only transform on test data to apply the SAME parameters, avoiding data leakage and ensuring consistent scaling.

---

## ✅ Quick Decision Checklist

- [ ] Is target variable categorical? → Use Logistic Regression
- [ ] Are there categorical columns? → Encode them appropriately
- [ ] Are there missing values? → Handle them (mean/mode/drop)
- [ ] Are there duplicates? → Remove them
- [ ] Did I split BEFORE scaling? → Avoid data leakage
- [ ] Did I use fit_transform on train only? → Prevent leakage
- [ ] Did I check confusion matrix? → Understand error types
- [ ] Did I print classification report? → See Precision/Recall/F1

---

## 📊 Metrics Quick Reference

| Scenario | Primary Metric | Secondary Metric |
|----------|----------------|------------------|
| Medical diagnosis | Recall | F1-Score |
| Spam filter | Precision | F1-Score |
| Fraud detection | Recall | Precision |
| Balanced classes | Accuracy | F1-Score |
| Imbalanced classes | F1-Score | Recall |
| Need single score | F1-Score | Accuracy |

---

## 🚫 Common Mistakes to Avoid

```
┌─────────────────────────────────────────────────────────────┐
│              COMMON MISTAKES                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ❌ Scaling before splitting → Data leakage                │
│  ❌ fit_transform on test data → Data leakage              │
│  ❌ Using same value for different categories in encoding  │
│  ❌ Forgetting to handle missing values                    │
│  ❌ Not checking target distribution (imbalanced data)     │
│  ❌ Using accuracy only for imbalanced datasets            │
│  ❌ Calling predict() before fit()                         │
│  ❌ Wrong parameter order: classification_report(y_pred,   │
│     y_test) instead of (y_test, y_pred)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
