# 🐼 Pandas Quick Reference Guide

## 📦 Import
```python
import pandas as pd
import numpy as np
```

---

## 1️⃣ Series
```python
# Create Series
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
```

---

## 2️⃣ DataFrame Creation
```python
data = {
    "Name": ["Ahmed", "Sara", None],
    "Age": [25, 30, 28],
    "City": ["Cairo", None, "Alex"]
}
df = pd.DataFrame(data)
```

> ⚠️ `"None"` (string) ≠ `None` (missing/NaN)

---

## 3️⃣ Data Exploration
| Method | Description |
|--------|-------------|
| `df.info()` | Structure, types, non-null counts |
| `df.head(n)` | First n rows |
| `df.tail(n)` | Last n rows |
| `df.describe()` | Numerical statistics |
| `df.describe(include='all').T` | All columns, transposed |

---

## 4️⃣ Selection & Filtering
```python
# Single column (Series)
df['Age']

# Multiple columns (DataFrame)
df[['Age', 'City']]

# Filter rows
df[df['Age'] > 25]

# Multiple conditions
df[(df['Age'] > 25) & (df['City'] == 'Cairo')]  # AND
df[(df['Age'] > 25) | (df['City'] == 'Cairo')]  # OR

# Label-based selection
df.loc[0:2, ['Age', 'City']]
```

---

## 5️⃣ Add & Drop Columns/Rows
```python
# Add column
df['Salary'] = [5000, 6000, 4500]

# Drop column
df.drop('Salary', axis=1)              # Returns new df
df.drop('Salary', axis=1, inplace=True) # Modifies original

# Drop row
df.drop(0, axis=0, inplace=True)

# Reset index
df.reset_index(drop=True)
```
> **axis=0** → Rows | **axis=1** → Columns

---

## 6️⃣ Missing Values
```python
# Check missing
df.isnull().sum()

# Fill missing
df['City'].fillna(df['City'].mode()[0], inplace=True)   # Categorical
df['Age'].fillna(df['Age'].mean(), inplace=True)        # Numerical
df['Salary'].fillna(df['Salary'].median(), inplace=True) # With outliers

# Drop rows with missing
df.dropna()
```

> 💡 Use `mode()[0]` to extract value from Series

---

## 7️⃣ GroupBy & Aggregation
```python
# Single aggregation
df.groupby('City')['Salary'].mean()
df.groupby('City')['Age'].max()

# Multiple aggregations
df.groupby('City').agg({'Salary': 'mean', 'Age': 'max'})
```

---

## 8️⃣ Apply Function
```python
# Regular function
def double(x):
    return x * 2

df['Double_Salary'] = df['Salary'].apply(double)

# Lambda
df['Double_Salary'] = df['Salary'].apply(lambda x: x * 2)

# Conditional function
def categorize(x):
    if x < 5000:
        return "Low"
    elif x <= 7000:
        return "Medium"
    else:
        return "High"

df['Category'] = df['Salary'].apply(categorize)
```

---

## 9️⃣ Sorting
```python
# Single column
df.sort_values('Salary', ascending=False)  # Highest first

# Multiple columns
df.sort_values(['Dept', 'Salary'], ascending=[True, False])
# Dept A→Z, then Salary high→low
```

| Order | ascending |
|-------|-----------|
| A → Z | `True` |
| Z → A | `False` |
| Low → High | `True` |
| High → Low | `False` |

---

## 🔟 Value Counts
```python
df['Department'].value_counts()
```

---

## 1️⃣1️⃣ Date Handling
```python
# Convert to datetime (SAVE IT!)
df['Date'] = pd.to_datetime(df['Date'])

# Filter by date
df[df['Date'] > "2020-01-01"]

# Extract components
df['Date'].dt.year
df['Date'].dt.month
df['Date'].dt.day_name()
```

---

## 1️⃣2️⃣ Read & Write CSV
```python
# Read
df = pd.read_csv("file.csv")

# Write
df.to_csv("output.csv", index=False)
```

---

## 📌 Common Mistakes to Avoid

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `df['col'] * 1.1` (for 10%) | `df['col'] * 0.1` |
| `df.mode()` | `df.mode()[0]` |
| `pd.to_datetime(df['Date'])` | `df['Date'] = pd.to_datetime(df['Date'])` |
| `df.drop('col', axis=1)` | `df.drop('col', axis=1, inplace=True)` or `df = df.drop()` |
| `df['A']` == `df[['A']]` | `df['A']` → Series, `df[['A']]` → DataFrame |

---

## 🧮 Quick Percentage Reference

| Percentage | Multiply by |
|------------|-------------|
| 10% | 0.10 |
| 15% | 0.15 |
| 20% | 0.20 |
| 110% (increase by 10%) | 1.10 |

---

## 🔗 Useful Patterns

```python
# Complete workflow
df = pd.read_csv("data.csv")           # Load
df.info()                               # Explore
df.isnull().sum()                       # Check missing
df['col'].fillna(df['col'].median(), inplace=True)  # Fill
df['new'] = df['old'].apply(func)      # Transform
df = df.sort_values('col', ascending=False)  # Sort
df.to_csv("output.csv", index=False)   # Save
```