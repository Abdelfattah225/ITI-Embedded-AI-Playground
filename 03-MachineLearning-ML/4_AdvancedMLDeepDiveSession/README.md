
# 📚 ML FUNDAMENTALS

## Quick Reference Guide for Review & Interviews

---

## 📌 1. SIGMOID FUNCTION

```
σ(z) = 1 / (1 + e^(-z))
```

| Property | Value |
|----------|-------|
| Output Range | (0, 1) |
| When z = 0 | 0.5 |
| When z → +∞ | 1 |
| When z → -∞ | 0 |

**Why use it?**
- Converts any number to probability (0-1)
- Linear regression output can be -∞ to +∞
- Sigmoid squashes it to valid probability

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

## 📌 2. LOG-LOSS (Binary Cross-Entropy)

```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

| Scenario | Loss |
|----------|------|
| Actual=1, Predict=0.99 | LOW ✅ |
| Actual=1, Predict=0.01 | HIGH ❌ |
| Actual=0, Predict=0.99 | VERY HIGH ❌❌ |

**Key insight:** Punishes **confident wrong** predictions SEVERELY!

```python
def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
```

---

## 📌 3. GRADIENT DESCENT

**Update Rule:**
```
w_new = w_old - α × gradient
```

| Learning Rate (α) | Effect |
|-------------------|--------|
| Too SMALL | Slow convergence |
| Just RIGHT | Smooth convergence |
| Too LARGE | Overshooting, divergence |

**Steps:**
1. Initialize random weights
2. Make predictions
3. Calculate loss
4. Calculate gradient
5. Update weights
6. Repeat!

---

## 📌 4. PRECISION & RECALL

```
              PREDICTED
            │ YES │ NO  │
      ──────┼─────┼─────┤
ACTUAL YES  │ TP  │ FN  │
      ──────┼─────┼─────┤
       NO   │ FP  │ TN  │
```

| Metric | Formula | Question |
|--------|---------|----------|
| **Precision** | TP/(TP+FP) | "Of predicted YES, how many correct?" |
| **Recall** | TP/(TP+FN) | "Of actual YES, how many found?" |
| **F1** | 2×P×R/(P+R) | Balance of both |

**When to prioritize:**
| Scenario | Prioritize | Why |
|----------|-----------|-----|
| Spam filter | Precision | Don't lose good emails |
| Cancer detection | Recall | Don't miss patients |
| Fraud detection | Recall | Don't miss fraudsters |

---

## 📌 5. R² SCORE

```
R² = 1 - (SS_res / SS_tot)
```

**Meaning:** % of variance explained by model

| R² Value | Interpretation |
|----------|----------------|
| 1.0 | Perfect |
| 0.8 | Good (80% explained) |
| 0.5 | Mediocre |
| 0.0 | Same as guessing mean |
| < 0 | Worse than guessing mean! |

---

## 📌 6. MODEL PROBLEMS

### Detection Table:

| Problem | Train Score | Test Score | Gap |
|---------|-------------|------------|-----|
| **Underfitting** | LOW | LOW | Small |
| **Good Fit** | HIGH | HIGH | Small |
| **Overfitting** | HIGH | LOW | LARGE |

### Data Leakage:
- **What:** Test info leaks into training
- **Common cause:** Scaling BEFORE split
- **Fix:** Always split FIRST, then scale

```python
# ❌ WRONG
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 📌 7. FIXING MODEL PROBLEMS

### Underfitting Solutions:
- Increase model complexity (higher degree)
- Add more features
- Reduce regularization
- Train longer

### Overfitting Solutions:
- Decrease model complexity
- Add regularization (Ridge/Lasso)
- Get more training data
- Early stopping

---

## 📌 8. REGULARIZATION

| Type | Penalty | Effect |
|------|---------|--------|
| **Ridge (L2)** | Σw² | Shrinks all weights |
| **Lasso (L1)** | Σ\|w\| | Can zero out weights |

**Alpha (α) controls strength:**
```
α = 0      → No regularization → Overfitting risk
α = small  → Light regularization
α = large  → Heavy regularization → Underfitting risk
```

```python
from sklearn.linear_model import Ridge, Lasso

model_ridge = Ridge(alpha=1.0)
model_lasso = Lasso(alpha=1.0)
```

---

## 📌 9. QUICK INTERVIEW ANSWERS

**Q: Explain overfitting in one sentence.**
> "Model memorizes training data instead of learning patterns, performing well on training but poorly on new data."

**Q: How to detect overfitting?**
> "High training score, low test score, large gap between them."

**Q: Precision vs Recall - when to use which?**
> "Precision when false positives are costly (spam filter). Recall when false negatives are costly (disease detection)."

**Q: What is data leakage?**
> "When information from test set influences training, giving unrealistically good results that won't work in production."

**Q: How does regularization prevent overfitting?**
> "Adds penalty for large weights, preventing model from fitting noise in training data."

**Q: What does R² = 0.85 mean?**
> "Model explains 85% of the variance in the target variable."

---

## 📌 10. COMMON MISTAKES TO AVOID

| Mistake | Correct Way |
|---------|-------------|
| Scale before split | Split first, then scale |
| fit_transform on test | Only transform on test |
| Too few test samples | Use at least 20% for test |
| Ignoring the gap | Always compare train vs test |
| Only looking at accuracy | Check precision, recall, F1 too |

---

## 📌 11. CODE TEMPLATE

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# 1. Load data
X, y = load_your_data()

# 2. Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale AFTER split
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LinearRegression()  # or Ridge, Lasso, LogisticRegression
model.fit(X_train_scaled, y_train)

# 5. Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Train: {train_score:.4f}")
print(f"Test:  {test_score:.4f}")
print(f"Gap:   {train_score - test_score:.4f}")

# 6. Diagnose
if train_score < 0.7 and test_score < 0.7:
    print("→ UNDERFITTING: Increase complexity")
elif train_score - test_score > 0.2:
    print("→ OVERFITTING: Add regularization")
else:
    print("→ GOOD FIT!")
```
