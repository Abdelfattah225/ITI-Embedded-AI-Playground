
# Smart Budget Analyzer & Fraud Detection System

A robust, menu-driven Python application designed to track financial transactions (Income/Expenses) while actively monitoring for suspicious activity. The system helps users manage their budget, view spending breakdowns, and automatically detects potential fraud based on spending patterns.

## 📌 Project Overview

This tool serves as a personal finance manager with built-in security features. It allows users to record transactions, validates inputs against a starting budget, and uses specific algorithms to flag high-value or repeated transactions that may indicate fraud.

**Key Features:**
*   **Budget Management:** Track income and expenses with real-time balance updates.
*   **Data Validation:** Prevents invalid entries (e.g., negative amounts, invalid categories, or expenses exceeding the current balance).
*   **Automated Fraud Detection:** Flags transactions over a specific monetary threshold or repeated transactions on the same day/category.
*   **Analytics:** View spending by category and generate monthly summaries.
*   **Preloaded Data:** The system initializes with sample data for immediate testing.

## 🛠 Tech Stack & Concepts

This project is built using **Python 3** and demonstrates proficiency in the following core programming concepts:

*   **Object-Oriented Programming (OOP):** Uses `Class` structures for `Transaction` and `BudgetTracker`.
*   **Data Structures:** Utilizes **Dictionaries** for categorization, **Sets** for unique constraints, **Tuples** for immutable data, and **Lists** for storage.
*   **Advanced Python Features:**
    *   List Comprehensions
    *   Tuple Unpacking
    *   Dictionary Manipulations
    *   Looping & Conditional Logic

## ⚙️ Configuration

 The system operates with the following default constants (customizable in the code):

*   **Starting Budget:** `5000`
*   **Fraud Alert Threshold:** `500`
*   **Allowed Categories:**
    *   Food
    *   Rent
    *   Transport
    *   Entertainment
    *   Health
    *   Other

## 🚀 How to Run

1.  Ensure you have Python installed on your machine.
2.  Save the script as `project.py`.
3.  Run the application via terminal/command prompt:

```bash
python project.py
```

## 📋 Usage Guide (Menu Options)

Upon launching, the system allows you to navigate via the following menu options:

### 1. Add Income / 2. Add Expense
Enter the Date (`DD-MM-YYYY`), Amount, Category, and Description.
*   *Note: Expenses greater than your current balance are rejected.*
*   *Note: Expenses greater than 500 will trigger a High Value Warning.*

### 3. View Transactions
Displays a raw list of all recorded transactions as tuples:
`(Date, Amount, Category, Type, Description)`

### 4. View Balance
Calculates the current balance based on: `Starting Budget (5000) + Total Income - Total Expenses`.

### 5. Spending by Category
Shows a dictionary summary of total expenses per category (e.g., `{'Food': 280, 'Rent': 600...}`).

### 6. Fraud Detection Report
Runs the analysis engine to display:
1.  **Suspicious Large Transactions:** Any expense over the 500 threshold.
2.  **Repeated Activity:** Identifies if multiple transactions occurred in the *same category* on the *same day*.

### 7. Monthly Summary
Input a specific Month (MM) and Year (YYYY) to see the total Income, Expense, and Net Savings for that period.

### 8. Unique Categories Used
Returns a set of all categories that have currently been used in your transaction history.

## 🔍 Fraud Detection Logic

The system utilizes two distinct logic checks to protect the user:

1.  **High-Value Detection:**
    *   Logic: `if type == "expense" and amount > THRESHOLD`
    *   Result: Added to a "Suspicious" list.

2.  **Frequency Analysis:**
    *   Logic: Tracks count of transactions grouped by `(Date, Category)`.
    *   Result: Flags specific dates and categories where activity occurred more than once (e.g., buying "Entertainment" twice on the same day).

## 📝 Example Output

```text
Initializing system with test data...
...Data pre-loaded successfully.


--- BUDGET ANALYZER MENU ---
1. Add Income
2. Add Expense
3. View Transactions
4. View Balance
5. Spending by Category
6. Fraud Detection Report
7. Monthly Summary
8. Unique Categories Used
9. Exit
Valid Categories: {'Rent', 'Other', 'Transport', 'Health', 'Food', 'Entertainment'}
FAILED: Transaction invalid. Check Category, negative amounts, or overdraft.

--- BUDGET ANALYZER MENU ---
1. Add Income
2. Add Expense
3. View Transactions
4. View Balance
5. Spending by Category
6. Fraud Detection Report
7. Monthly Summary
8. Unique Categories Used
9. Exit

Current Balance: $8790

--- BUDGET ANALYZER MENU ---
1. Add Income
2. Add Expense
3. View Transactions
4. View Balance
5. Spending by Category
6. Fraud Detection Report
7. Monthly Summary
8. Unique Categories Used
9. Exit

Current Balance: $8790

--- BUDGET ANALYZER MENU ---
1. Add Income
2. Add Expense
3. View Transactions
4. View Balance
5. Spending by Category
6. Fraud Detection Report
7. Monthly Summary
8. Unique Categories Used
9. Exit
Exiting system. Goodbye!```
