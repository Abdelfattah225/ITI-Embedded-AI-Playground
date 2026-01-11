
# 🌳 Decision Tree - Cheat Sheet

Quick reference guide for Data Science & ML interviews.

---

## 1️⃣ What is Decision Tree?

> A supervised learning algorithm that makes predictions by learning decision rules from features, creating a tree-like flowchart structure.

**Used for:** Classification & Regression

---

## 2️⃣ Key Terminology

| Term | Definition |
|------|------------|
| Root Node | First node (top), best feature split |
| Internal Node | Decision points with children |
| Leaf Node | Terminal node with final prediction |
| Depth | Longest path from root to leaf |
| Pruning | Cutting branches to reduce overfitting |

---

## 3️⃣ Core Formulas

### Entropy
```
Entropy = -Σ pi × log₂(pi)

• 0 = Pure (all same class)
• 1 = Maximum impurity (50-50 split)
```

### Gini Impurity
```
Gini = 1 - Σ pi²

• 0 = Pure
• 0.5 = Maximum impurity
```

### Information Gain
```
IG = Entropy(Parent) - Weighted Avg Entropy(Children)

• Higher IG = Better split
• Feature with highest IG → Root Node
```

---

## 4️⃣ How Tree Selects Root Node?

1. Calculate Information Gain for ALL features
2. Select feature with **HIGHEST** Information Gain
3. That feature becomes root node
4. Repeat process for child nodes

---

## 5️⃣ Entropy vs Gini

| Aspect | Entropy | Gini |
|--------|---------|------|
| Formula | Uses log | Uses squares |
| Speed | Slower | Faster |
| Range | 0 to 1 | 0 to 0.5 |
| Sklearn Default | No | **Yes** |

---

## 6️⃣ Overfitting

### Signs:
- Training Accuracy: 98%
- Test Accuracy: 60%
- **Large gap = Overfitting!**

### Solutions:

| Parameter | Change | Effect |
|-----------|--------|--------|
| max_depth | ↓ Decrease | Shorter tree |
| min_samples_split | ↑ Increase | Fewer splits |
| min_samples_leaf | ↑ Increase | Bigger leaves |
| max_leaf_nodes | ↓ Decrease | Fewer leaves |

---

## 7️⃣ Pruning Types

| Type | When | How |
|------|------|-----|
| Pre-Pruning | Before building | Set max_depth, min_samples |
| Post-Pruning | After building | Remove unhelpful branches |

---

## 8️⃣ Hyperparameters

```python
DecisionTreeClassifier(
    criterion='gini',      # 'gini' or 'entropy'
    max_depth=5,           # Limit depth
    min_samples_split=10,  # Min samples to split
    min_samples_leaf=5,    # Min samples in leaf
    max_leaf_nodes=20      # Max leaves
)
```

---

## 9️⃣ Advantages & Disadvantages

| ✅ Pros | ❌ Cons |
|---------|---------|
| Easy to interpret | Overfitting prone |
| No feature scaling | Unstable |
| Handles all data types | Greedy (not optimal) |
| Shows feature importance | Biased with imbalanced data |
| Fast prediction | Single tree = less accurate |

---

## 🔟 Common Interview Questions

### Q1: How does Decision Tree select the best feature?
> By calculating Information Gain for all features and selecting the one with highest IG (reduces impurity most).

### Q2: Difference between Gini and Entropy?
> Both measure impurity. Gini uses squares (faster), Entropy uses log. Results are similar.

### Q3: How to prevent overfitting?
> Use pruning: limit max_depth, increase min_samples_split/leaf, limit max_leaf_nodes.

### Q4: Why is it called "greedy" algorithm?
> Makes best split at each step without considering future splits. May not find global optimum.

### Q5: When to use Decision Tree?
> When interpretability matters, data has non-linear relationships, mixed feature types.

### Q6: Decision Tree vs Random Forest?
> Random Forest = ensemble of many Decision Trees. More accurate but less interpretable.

---

## 📝 Quick Code

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(accuracy_score(y_test, y_pred))

# Feature Importance
print(model.feature_importances_)
```

---

## 🎯 One-Liner Definitions

| Concept | One-Liner |
|---------|-----------|
| Decision Tree | Tree-based model that splits data using feature rules |
| Entropy | Measure of randomness/impurity in data |
| Gini | Alternative impurity measure using squared probabilities |
| Information Gain | Reduction in impurity after split |
| Pruning | Cutting tree branches to prevent overfitting |
| Overfitting | Model memorizes training data, fails on new data |

---

## ⚡ Last-Minute Revision

```
REMEMBER:
├── Impurity: Entropy & Gini
├── Split by: Highest Information Gain
├── Overfitting: Big gap Train vs Test
├── Fix Overfitting: ↓max_depth, ↑min_samples
├── Sklearn default: criterion='gini'
└── Tree is GREEDY (local optimal, not global)
```

---

Good luck with your interview! 🚀
```

---

# ✅ README Complete!

| Section | Covered |
|---------|---------|
| Definition | ✅ |
| Terminology | ✅ |
| Formulas | ✅ |
| How it works | ✅ |
| Overfitting | ✅ |
| Hyperparameters | ✅ |
| Pros/Cons | ✅ |
| Interview Q&A | ✅ |
| Quick Code | ✅ |
| One-liners | ✅ |

---

# 🎯 Ready for Naive Bayes?

Now let's continue! Answer the 3 tasks:

**Task 1:** Match: 1-?, 2-?, 3-?, 4-?
```
1. Prior      A. P(B|A)
2. Likelihood B. P(A|B)  
3. Posterior  C. P(A)
4. Evidence   D. P(B)
```

**Task 2:** Calculate P(PASS | Studied)

**Task 3:** If P(SPAM|FREE) = 80.4%, what is P(NOT SPAM|FREE)?