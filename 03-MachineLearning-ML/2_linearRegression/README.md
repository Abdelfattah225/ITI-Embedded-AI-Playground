
# Linear Regression - Complete Guide

## 📌 What is Linear Regression?

Linear Regression predicts **continuous numerical values** based on input features.

### Formula
```
Simple:      y = wx + b
Multivariate: y = w₁x₁ + w₂x₂ + w₃x₃ + ... + b

Where:
- y = predicted value
- x = input features
- w = weights (coefficients)
- b = bias (intercept)
```

---

## 🎯 When to Use Linear Regression?

| Use Linear Regression | DON'T Use |
|-----------------------|-----------|
| House price prediction | Yes/No questions |
| Salary prediction | Spam detection |
| Temperature forecast | Disease diagnosis |
| Sales forecasting | Image classification |

**Rule:** Target variable must be **continuous numerical**

---

## 📊 Complete Pipeline

```
1. Load Data
2. EDA (Exploratory Data Analysis)
3. Encode Categorical Variables
4. Split Data (train/test)
5. Scale Numerical Features
6. Train Model
7. Evaluate
```

⚠️ **Important:** Split BEFORE Scaling to avoid data leakage!

---

## 🔤 Encoding Types

| Type | When to Use | Example |
|------|-------------|---------|
| Binary | 2 categories | Yes/No → 1/0 |
| One-Hot | Multiple categories, NO order | Colors → [1,0,0], [0,1,0] |
| Ordinal | Multiple categories WITH order | Low=1, Medium=2, High=3 |

---

## 📏 Scaling (StandardScaler)

**Why?** Features with different ranges affect model unfairly.

**What it does:**
- Mean = 0
- Standard Deviation = 1

**Code:**
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Learn + Apply
x_test = scaler.transform(x_test)         # Only Apply
```

⚠️ **Never** use `fit_transform` on test data!

---

## 🚫 Data Leakage

**Definition:** Model accidentally sees test data during training.

**How to Avoid:**
- Split data BEFORE scaling
- Use `transform()` (not `fit_transform`) on test data
- Keep test data completely separate

**Result of Leakage:** Fake high accuracy that won't work in real world.

---

## 📈 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| MAE | avg(\|actual - predicted\|) | Average error in original units |
| MSE | avg((actual - predicted)²) | Penalizes large errors more |
| R² | 0 to 1 | % of variance explained (higher = better) |

---

## 🔑 Interpreting Coefficients

```python
# After training
model.coef_       # Weights for each feature
model.intercept_  # Bias (base value)
```

**Example:**
```
Coefficients: [50, 10000, -200]
Features:     [area, bedrooms, age]

Meaning:
- +1 sqft area → +$50 price
- +1 bedroom → +$10,000 price
- +1 year age → -$200 price (negative = inverse relationship)
```

---

## 💻 Complete Code Template

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv("data.csv")

# 2. EDA
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()

# 3. Handle Missing Values (if any)
df['column'].fillna(df['column'].mean(), inplace=True)  # numerical
df['column'].fillna(df['column'].mode()[0], inplace=True)  # categorical

# 4. Encode Categorical Variables
# Binary
df['yes_no_col'] = df['yes_no_col'].replace({'yes': 1, 'no': 0})

# One-Hot
df = pd.get_dummies(df, columns=['category_col'])

# 5. Separate Features and Target
x = df.drop('target', axis=1)
y = df['target']

# 6. Split Data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=42
)

# 7. Scale Features
scaler = StandardScaler()
numerical_cols = ['col1', 'col2', 'col3']

x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])

# 8. Train Model
model = LinearRegression()
model.fit(x_train, y_train)

# 9. Predict
y_pred = model.predict(x_test)

# 10. Evaluate
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# 11. Check What Model Learned
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

---

## ❓ Common Interview Questions

### Q1: What is the difference between MAE and MSE?
**A:** MAE treats all errors equally. MSE squares errors, so it penalizes large errors more heavily.

### Q2: What is data leakage?
**A:** When test data information is accidentally used during training, giving fake high accuracy.

### Q3: Why do we scale features?
**A:** To put all features on the same scale so no single feature dominates due to larger numbers.

### Q4: Why split before scaling?
**A:** To prevent data leakage. Scaling should only learn from training data.

### Q5: What does a negative coefficient mean?
**A:** The feature has an inverse relationship with target. When feature increases, target decreases.

### Q6: What is R² score?
**A:** Percentage of variance in target variable explained by the model. Range 0-1, higher is better.

### Q7: What is overfitting?
**A:** Model memorizes training data instead of learning patterns. High training accuracy, low test accuracy.

---

## 🎯 Quick Decision Checklist

- [ ] Is target variable continuous? → Use Linear Regression
- [ ] Are there categorical columns? → Encode them
- [ ] Did I split BEFORE scaling? → Avoid data leakage
- [ ] Did I use fit_transform on train only? → Prevent leakage
- [ ] Did I call model.fit() before predict()? → Required!

---

## 🔄 Pipeline Summary Diagram

```
Data → EDA → Encode → Split → Scale → Train → Predict → Evaluate
                        ↓
              fit_transform(train)
              transform(test)
```

