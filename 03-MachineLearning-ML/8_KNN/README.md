# 📊 K-Nearest Neighbors (KNN)

## 🎯 What is KNN?

KNN is a simple, intuitive machine learning algorithm that classifies data points based on their **nearest neighbors**. The idea: *"Tell me who your neighbors are, and I'll tell you who you are!"*

## 🔍 How It Works

1. **Store** all training data (no actual training!)
2. When a new point arrives:
   - Calculate distance to all training points
   - Find K nearest neighbors
   - Take majority vote
   - Return prediction

```
Example: K=3

    🔴 No
  🔵    🔴 No
    ⭐ NEW
  🔵  🔵 Yes
    🔵 Yes

Vote: 3 Yes, 1 No → Prediction: Yes ✅
```

## 💻 Quick Start

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale features (IMPORTANT!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Create and train KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled, y_train)

# Predict
prediction = knn.predict(scaler.transform(X_test))
```

## ⚙️ Key Parameters

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `n_neighbors` | Number of neighbors (K) | 3, 5, 7 |
| `metric` | Distance calculation | `'euclidean'`, `'manhattan'` |
| `weights` | Voting method | `'uniform'`, `'distance'` |

## 📏 Distance Metrics

- **Euclidean**: Straight-line distance `√[(x₁-x₂)² + (y₁-y₂)²]`
- **Manhattan**: City-block distance `|x₁-x₂| + |y₁-y₂|`
- **Minkowski**: Generalized distance (p=1: Manhattan, p=2: Euclidean)

## ✅ Advantages

- ✨ Simple and intuitive
- 📚 No training required
- 🎯 Works for classification & regression
- 🔄 Easily adapts to new data

## ❌ Disadvantages

- 🐌 Slow prediction (compares to all points)
- 💾 High memory usage (stores all data)
- 📐 Sensitive to feature scales
- 🚫 Poor for high-dimensional data
- ⚡ Not suitable for embedded systems

## ⚠️ Important: Feature Scaling!

**Always scale features before using KNN!**

```python
# Without scaling: Large features dominate
Age: 25-50 (range: 25)
Salary: 20k-90k (range: 70k) ← Dominates distance!

# Solution: StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 🎯 Choosing K

**Rule of thumb:** K = √(number of samples)

- **Small K (1-3)**: Sensitive to noise, may overfit
- **Medium K (5-10)**: Good balance
- **Large K**: Smoother boundaries, may underfit

**Use Elbow Method** to find optimal K!

## 🔧 Best Practices

1. **Scale your features** using `StandardScaler`
2. **Try different K values** (use cross-validation)
3. **Experiment with distance metrics**
4. **Use `weights='distance'`** for better accuracy
5. **Remove irrelevant features** (curse of dimensionality)

## 📊 When to Use KNN?

| ✅ Good For | ❌ Avoid For |
|------------|-------------|
| Small datasets | Large datasets |
| Non-linear patterns | Real-time applications |
| Quick prototyping | Embedded systems |
| Low-dimensional data | High-dimensional data |

## 🚀 Example: Complete Pipeline

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Grid search for best parameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

# 4. Best model
best_knn = grid.best_estimator_
print(f"Best params: {grid.best_params_}")
print(f"Test accuracy: {accuracy_score(y_test, best_knn.predict(X_test_scaled)):.3f}")
```


## 🎓 Summary

KNN is a **lazy learner** that makes predictions by voting among K nearest neighbors. While simple and effective for small datasets, it's **not recommended for production or embedded systems** due to slow prediction and high memory requirements.

**Key Formula:**
```
For new point X:
1. Calculate distance to all training points
2. Select K nearest neighbors
3. Prediction = Majority vote (or weighted average)
```

---

⭐ **Remember:** Always scale your features! It's the most common mistake with KNN.
```