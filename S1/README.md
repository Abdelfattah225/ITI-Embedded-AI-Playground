# Python Data Structures Guide

A quick reference guide to Python's core data structures: Lists, Tuples, Sets, and Dictionaries.

---

## 📋 Lists

**Ordered, mutable collection that allows duplicates**

```python
fruits = ["apple", "banana", "cherry"]
```

### Characteristics
- Defined with square brackets `[]`
- Can be modified after creation
- Accessed by index (starting at 0)
- Allows duplicate values

### Common Operations
- `list.append(item)` - Add item to end
- `list.insert(index, item)` - Insert at position
- `list.remove(item)` - Remove by value
- `list.pop()` - Remove and return last item
- `list.sort()` - Sort in place
- `len(list)` - Get length

### Use Case
Dynamic collections that need frequent modifications

---

## 🔒 Tuples

**Ordered, immutable collection that allows duplicates**

```python
coordinates = (10, 20, 30)
```

### Characteristics
- Defined with parentheses `()`
- Cannot be modified after creation
- Accessed by index
- Faster than lists

### Common Operations
- `tuple.count(item)` - Count occurrences
- `tuple.index(item)` - Find index
- `len(tuple)` - Get length
- Tuple unpacking: `x, y = (1, 2)`

### Use Case
Fixed data that shouldn't change, function return values, dictionary keys

---

## 🎯 Sets

**Unordered collection of unique items**

```python
unique_numbers = {1, 2, 3, 4, 5}
```

### Characteristics
- Defined with curly braces `{}`
- Automatically removes duplicates
- No indexing or order
- Fast membership testing

### Common Operations
- `set.add(item)` - Add single item
- `set.remove(item)` - Remove item
- `set1 | set2` - Union
- `set1 & set2` - Intersection
- `set1 - set2` - Difference
- `item in set` - Very fast membership test

### Use Case
Remove duplicates, fast lookups, mathematical set operations

---

## 📖 Dictionary

**Unordered collection of key-value pairs**

```python
person = {"name": "Alice", "age": 25, "city": "NYC"}
```

### Characteristics
- Defined with `{key: value}` pairs
- Keys must be unique and immutable
- Values can be any type
- Ordered (Python 3.7+)

### Common Operations
- `dict[key]` - Access value
- `dict.get(key, default)` - Safe access
- `dict[key] = value` - Add/update
- `dict.pop(key)` - Remove and return
- `dict.keys()` - Get all keys
- `dict.values()` - Get all values
- `dict.items()` - Get key-value pairs

### Use Case
Storing structured data, configurations, fast key-based lookups

---

## 📊 Quick Comparison

| Feature | List | Tuple | Set | Dictionary |
|---------|------|-------|-----|------------|
| **Syntax** | `[]` | `()` | `{}` | `{k: v}` |
| **Ordered** | ✅ | ✅ | ❌ | ✅ (3.7+) |
| **Mutable** | ✅ | ❌ | ✅ | ✅ |
| **Duplicates** | ✅ | ✅ | ❌ | ❌ (keys) |
| **Indexed** | ✅ | ✅ | ❌ | By key |
| **Speed** | Medium | Fast | Very Fast | Very Fast |

---

## 💡 When to Use What?

- **List**: Need ordered collection with modifications
- **Tuple**: Need ordered collection that won't change
- **Set**: Need unique items or fast membership testing
- **Dictionary**: Need to map keys to values

---

## 🚀 Quick Tips

1. Use `tuple` for data that shouldn't change
2. Use `set` to remove duplicates from a list: `list(set(my_list))`
3. Use `dict.get()` instead of `dict[key]` to avoid KeyErrors
4. Lists and dictionaries are most commonly used in everyday Python
5. Remember: Empty set must be `set()`, not `{}` (which creates a dict)

---
