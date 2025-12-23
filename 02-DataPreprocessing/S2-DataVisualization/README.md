# Matplotlib & Seaborn Quick Reference

---

## 📦 Import
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 1️⃣ Line Plot
**Use:** Trends over time / continuous data

```python
# Matplotlib
plt.plot(x, y)
plt.title("Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()

# Seaborn
sns.lineplot(x='col1', y='col2', data=df)
```

---

## 2️⃣ Scatter Plot
**Use:** Relationship between 2 variables

```python
# Matplotlib
plt.scatter(x, y)

# Seaborn
sns.scatterplot(x='col1', y='col2', data=df)

# With color grouping
sns.scatterplot(x='col1', y='col2', data=df, hue='category')
```

---

## 3️⃣ Bar Chart
**Use:** Compare categorical data

```python
# Matplotlib
plt.bar(categories, values, color='green')

# Seaborn (shows MEAN by default)
sns.barplot(x='category', y='value', data=df)

# Seaborn with SUM
sns.barplot(x='category', y='value', data=df, estimator=sum)
```

---

## 4️⃣ Histogram
**Use:** Distribution of numerical data

```python
# Matplotlib
plt.hist(data, bins=10, color='blue')

# Seaborn (with density curve)
sns.histplot(x=df['column'], bins=20, kde=True)
```

### Bins Parameter:
| bins | Result |
|------|--------|
| Small (5) | Few wide bars, less detail |
| Large (50) | Many narrow bars, more detail |

---

## 5️⃣ Count Plot
**Use:** Count occurrences of categories

```python
sns.countplot(x='category', data=df)
```

### barplot vs countplot:
| Plot | Purpose |
|------|---------|
| `barplot` | Mean/Sum of values |
| `countplot` | Count of occurrences |

---

## 6️⃣ Box Plot
**Use:** Distribution, median, outliers

```python
sns.boxplot(x='category', y='value', data=df)
```

```
    ┌─────┐ ← Maximum
    │     │
    │──── │ ← Median (Q2)
    │     │
    └─────┘ ← Q1 and Q3
       •    ← Outliers
```

---

## 7️⃣ Heatmap
**Use:** Correlation matrix visualization

```python
# Select numerical columns only
numerical = df.select_dtypes(['int64', 'float64'])

# Create heatmap
sns.heatmap(numerical.corr(), annot=True, cmap='coolwarm')
```

### Parameters:
| Parameter | Purpose |
|-----------|---------|
| `annot=True` | Show numbers |
| `cmap='Blues'` | Color scheme |

### Correlation Values:
| Value | Meaning |
|-------|---------|
| +1.0 | Strong positive |
| 0.0 | No correlation |
| -1.0 | Strong negative |

---

## 8️⃣ Subplots
**Use:** Multiple plots in one figure

```python
# Method 1: plt.subplot(rows, cols, position)
plt.subplot(2, 2, 1)  # 2 rows, 2 cols, position 1
plt.plot(x, y)
plt.title("Plot 1")

plt.subplot(2, 2, 2)
plt.scatter(x, y)
plt.title("Plot 2")

plt.tight_layout()
plt.show()
```

### Layout Guide:
```
subplot(2,2,1) │ subplot(2,2,2)
───────────────┼───────────────
subplot(2,2,3) │ subplot(2,2,4)
```

---

## 9️⃣ The `hue` Parameter
**Use:** Add color grouping by category

```python
# Adds third dimension with colors
sns.barplot(x='day', y='total_bill', data=df, hue='sex')
sns.scatterplot(x='x', y='y', data=df, hue='category')
```

---

## 🔟 Common Colormaps
| cmap | Best for |
|------|----------|
| `'Blues'` | Sequential data |
| `'coolwarm'` | Diverging (+ and -) |
| `'viridis'` | General purpose |
| `'RdYlGn'` | Good/Bad comparison |

---

## 📊 Built-in Seaborn Datasets
```python
df = sns.load_dataset("tips")
df = sns.load_dataset("flights")
df = sns.load_dataset("iris")
df = sns.load_dataset("titanic")
```

---

## 📌 Quick Reference Table

| Plot Type | Matplotlib | Seaborn |
|-----------|------------|---------|
| Line | `plt.plot()` | `sns.lineplot()` |
| Scatter | `plt.scatter()` | `sns.scatterplot()` |
| Bar | `plt.bar()` | `sns.barplot()` |
| Histogram | `plt.hist()` | `sns.histplot()` |
| Box | - | `sns.boxplot()` |
| Count | - | `sns.countplot()` |
| Heatmap | - | `sns.heatmap()` |

---

## ⚠️ Common Mistakes

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `plt.plot(y, x)` | `plt.plot(x, y)` - X first! |
| `bins` controls gaps | `bins` controls number of bars |
| `plt.bar()` for distribution | `plt.hist()` for distribution |
| Heatmap on all columns | Heatmap on numerical only |

---

## 🔗 Complete Workflow

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = sns.load_dataset("tips")

# Quick exploration
df.head()
df.info()

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=df, hue='time')
plt.title("Tips vs Total Bill")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.tight_layout()
plt.show()

# Save plot
plt.savefig("my_plot.png")
```

---

## 📐 Figure Size
```python
plt.figure(figsize=(width, height))
# Example
plt.figure(figsize=(12, 6))
```

---

## 💾 Save Plot
```python
plt.savefig("plot.png")
plt.savefig("plot.pdf")
plt.savefig("plot.png", dpi=300)  # High resolution
```
