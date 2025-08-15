# WEEK 1: Introduction to Neural Networks

## Neural Network Basics

- Simplest network: `x1 → network → f(x)`
- Network is **automatically formed** to give required output
- We are in the field of **Supervised Learning** (i.e., input-output pairs are given)

---

## Deep Learning - Vectorized Neural Network Forward Propagation 

### 1. Logistic Regression Recap (Single Example)

- Input feature: $x \in \mathbb{R}^{n_x}$  
  Example: all RGB image pixels converted into a single vector
- Output: $y \in \{0, 1\}$  
  Example: cat or not-cat
- Training examples (m): ${(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots}$

To handle multiple examples:

- $X \in \mathbb{R}^{n_x \times m}$
- $Z = w^T X + b$ → linear combination
- $A = \sigma(Z)$ → activation function to get probabilities (sigmoid)

### 2. Logistic Regression Cost Function

- Goal: predicted output $\hat{y} \approx y$
- If $y = 1$: only $\log(\hat{y})$ matters → more $\hat{y} \Rightarrow$ less loss
- Use **cross-entropy loss**

### 3. Gradient Descent

We minimize cost function $J(w, b)$:

$$
w = w - \alpha \cdot \frac{\partial J}{\partial w} \\
b = b - \alpha \cdot \frac{\partial J}{\partial b}
$$

- Gradient points toward steepest increase → we move in **opposite** direction
- Step size is controlled by $\alpha$ (learning rate)

### 4. Vectorization

- Naive way: use loops to compute $z = wx + b$
- Efficient way: use NumPy vectorization:

```python
z = np.dot(w, x) + b
```

- Removes need for loops; works on entire batch at once

### 5. Neural Network Representation (One Hidden Layer)

We build on logistic regression by adding hidden units:

- Hidden layer: 4 neurons
- Output layer: 1 neuron

Each hidden unit:

- $z_j^{[1]} = w_j^{[1]^T} x + b_j^{[1]}$
- $a_j^{[1]} = \sigma(z_j^{[1]})$

Matrix form:

- $Z^{[1]} = W^{[1]} x + b^{[1]}$, $A^{[1]} = \sigma(Z^{[1]})$
- $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$, $A^{[2]} = \sigma(Z^{[2]}) = \hat{y}$

### 6. Backward Propagation (Gradient Descent Continued)

- Update all weights and biases by calculating gradients from cost
- Starts by assuming all other weights and biases are already optimized (shown in green).
- The network outputs a **"green"** — the predicted curve from the model.
- `b` shifts the squiggle **vertically** to better fit the data.

---

## Using the Chain Rule

- Compute derivative of loss w.r.t. `b`:

```
d(loss)/db3 = d(loss)/dŷ × d(ŷ)/db3
```

- Shows step-by-step how each part is derived.

---

## Gradient Descent Process

1. Start with `b = 0`
2. Compute the slope using derivative of loss
3. Calculate new `b`:
   ```
   b = b - learning_rate × slope
   ```
4. Repeat until step size ≈ 0


### 7. Random Initialization

- Required to break symmetry between neurons if so not done we see learning stuck in a loop or stays incomplete 

### 8. Key Insights

- Activations: $A^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$
- Vectorization improves speed
- Broadcasting applies bias across all examples

### 9. Forward vs Backward Propagation

| Step       | Forward Propagation          | Backward Propagation           |
| ---------- | ---------------------------- | ------------------------------ |
| Purpose    | Predict output $hat{y}$      | Update weights from error      |
| Flow       | Input → Hidden → Output      | Output → Hidden → Input        |
| Use Case   | Every prediction or training | Only during training           |
| Involves   | $z, a = \sigma(z)$           | Gradients via chain rule       |
| In project | Convert voice to command     | Learn to correct wrong outputs |

### 10. Activation Functions

| Function | Formula                                  | Output  | Use Case              |
| -------- | -----------------------------------------| ------- | --------------------- |
| Sigmoid  | $frac{1}{1 + e^{-x}}$                    | (0, 1)  | Binary classification |
| ReLU     | $max(0, x)$                              | [0, ∞)  | Hidden layers         |
| Tanh     | $tanh(x)$                                | (-1, 1) | Centered around zero  |
| Softmax  | $frac{e^{z_i}}{\sum_j e^{z_j}}$          | [0, 1]  | Multi-class output    |

### 11. Why Non-Linearity?

Without non-linear functions:

- Even deep networks behave like linear models
- Non-linearity is **essential** to capture complex real-world patterns

### 12. Loss Function (Binary Classification)

$$
\mathcal{L}(\hat{y}, y) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

Total cost over m examples:

$$
J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
$$

### 13. Backward Propagation: Vectorized Update

```python
dZ[l] = dA[l] * activation_derivative(Z[l])
dW[l] = (1/m) * np.dot(dZ[l], A[l-1].T)
db[l] = (1/m) * np.sum(dZ[l], axis=1, keepdims=True)
dA[l-1] = np.dot(W[l].T, dZ[l])
```

### 14. Real World: Voice to Engine Mapping

| Component          | Role                                        |
| ------------------ | ------------------------------------------- |
| Forward Prop       | Predicts command from your voice            |
| Backward Prop      | Trains system to correct errors             |
| ReLU/Hidden Layers | Learn complex patterns in voice features    |
| Sigmoid/Softmax    | Final decision-making                       |
| Non-linearity      | Needed to map real-world audio to decisions |

---



We want a little network to look at a number and decide:
**Is it more like a 1 or more like a 0?**

The network uses a tiny neural network with:

- 1 input (`x`)
- 1 weight (`w`)
- 1 bias (`b`)
- 1 output (`ŷ`) using the **sigmoid** function

---

## 1: Forward Pass (network Makes a Guess)

Given:

- Input: `x = 2`
- Weight: `w = 1`
- Bias: `b = 0`

The network does:

```
z = w * x + b = 1 * 2 + 0 = 2
ŷ = sigmoid(z) ≈ 0.88
```

So the network predicts it's probably a **1**.

---

## 2: Calculate Loss (How Wrong Was the network?)

True label: `y = 1`\
We use binary cross-entropy:

```
Loss = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]
     = -[1 * log(0.88)] ≈ 0.13
```

Small loss = network did pretty well!

---

## 3: Backward Pass (network Learns)

The network asks:

> "How should I change my weight or bias to do better next time?"

We compute:

- `dLoss/dw`: How much did the **weight** affect the loss?
- `dLoss/db`: How much did the **bias** affect the loss?

These come from **derivatives** and **chain rule**.

---

##   4: Update the Weights (Gradient Descent)

The network updates:

```
w = w - learning_rate * dLoss/dw
b = b - learning_rate * dLoss/db
```

Let’s say it updates:

- `w` from `1` → `0.98`\
  Now it will do slightly better next time.

---

## Summary

> The network guesses → sees how wrong it is → figures out what caused the mistake → fixes the mistake → guesses better next time!

