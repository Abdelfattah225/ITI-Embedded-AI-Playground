# Machine Learning: Deep Dive Notes

## 1. Foundations
*   **Machine Learning:** The science of getting computers to act without explicit programming.
    *   *Traditional:* Input + Rules $\rightarrow$ Output
    *   *ML:* Input + Output $\rightarrow$ Rules
*   **Modeling:** Approximating real-world relationships.
    *   *Real World:* $y = f(x) + \text{noise}$
    *   *Our Model:* $\hat{y} = h(x)$ (Hypothesis)
*   **ML vs. Deep Learning:**
    *   *ML (Shallow):* One function mapping ($y = f(x)$). Requires manual feature extraction.
    *   *DL (Deep):* Multiple composite functions ($y = f(g(h(x)))$). Learns features automatically via layers.

## 2. Linear Regression Mechanics
**Equation:** $y = wx + b$
*   **$w$ (Weight):** Importance of the feature (Slope).
*   **$b$ (Bias):** The starting point when input is 0 (Intercept). Allows the line to shift up/down.

### The Cost Function (MSE)
**Mean Squared Error:** $\frac{1}{n} \sum (y - \hat{y})^2$
*   *Why Square?*
    1.  Eliminates negatives (so errors don't cancel out).
    2.  **Punishes** large errors more severely than small ones.

### Optimization: Gradient Descent
Walking down the "error mountain" to find the lowest MSE.
*   **Update Rule:** $w_{new} = w_{old} - (\text{LearningRate} \times \text{Gradient})$
*   **Learning Rate:** Step size. Too big = Overshoot/Diverge. Too small = Slow.

## 3. Training Concepts
*   **Parameters ($w, b$):** Internal variables learned by the model *during* training.
*   **Hyperparameters:** External settings configured by the human *before* training (e.g., Learning Rate, Epochs, Batch Size).
*   **Epoch:** One complete pass through the entire dataset.
*   **Parameter Grid:** A method to test combinations of hyperparameters to find the best settings.

## 4. Optimization Variations
1.  **Batch Gradient Descent:** Calculates error on **ALL** data before taking 1 step. (Accurate but slow).
2.  **Stochastic GD:** Calculates error on **1** row, takes 1 step. (Fast but noisy/zig-zag path).
3.  **Mini-Batch GD:** The industry standard. Uses small chunks (e.g., 32, 64 rows). Balances speed and stability.

## 5. Model Performance (Generalization)
The goal is the "Goldilocks Zone."

### Underfitting (High Bias)
*   **Description:** Model is too simple to capture the pattern.
*   **Symptoms:** High Training Error, High Test Error.
*   **Fix:** Make model more complex, add features, train longer.

### Overfitting (High Variance)
*   **Description:** Model memorizes noise instead of patterns.
*   **Symptoms:** Low Training Error, High Test Error.
*   **Fix:** Add more data, Feature selection, **Regularization**.
