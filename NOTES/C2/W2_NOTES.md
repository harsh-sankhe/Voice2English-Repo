# WEEK2
## Batch Gradient Descent
- Training neural networks involves trying many models to find one that works well, which is an empirical process (testing and experimenting to see what works). Faster training helps you test more models quickly.
Large datasets (Big Data) make training slow because processing all examples at once takes a long time.
  1. Batch Gradient Descent:
   -  In regular (batch) gradient descent, you process the entire dataset (e.g., 5 million examples) to compute one gradient and take  one step to update the model’s parameters (weights and biases).
   - This is slow because you must wait to process all examples before updating the model once.
  2. Mini-Batch Gradient Descent:
   - Instead of using the whole dataset, you split it into smaller chunks called mini-batches (e.g., 1,000 examples each).
     For a dataset with 5 million examples, splitting into mini-batches of 1,000 gives you 5,000 mini-batches.
   - You process one mini-batch at a time, compute the gradient, and update the model. This lets you take many small steps (e.g., 5,000 steps per pass through the data) instead of one big step.

- How Mini-Batch Gradient Descent Works:
  - For each mini-batch (denoted $ X^{\{t\}}, Y^{\{t\}} $, where $ t $ is the mini-batch index):
    - Forward Propagation: Compute predictions using the mini-batch inputs $ X^{\{t\}} $.
    - Loss Calculation: Compute the loss for the mini-batch, including any regularization (a penalty to prevent overfitting).
    - Backpropagation: Calculate gradients for the mini-batch.
    - Update Parameters: Adjust weights and biases using the gradient.
  - One complete pass through all mini-batches (e.g., 5,000 mini-batches) is called an epoch (one full cycle through the training data).
You repeat multiple epochs until the model learns well.


- Why It’s Faster:
Mini-batch gradient descent allows you to update the model after processing just 1,000 examples (or another small batch size), so you make progress without waiting for the entire dataset.
Vectorization (processing multiple examples at once using matrix operations) makes each mini-batch computation efficient, even though it’s smaller than the full dataset.

- Notation:
The dataset is split into mini-batches: $ X^{\{t\}} $ (inputs) and $ Y^{\{t\}} $ (outputs), where $ t $ is the mini-batch number (e.g., $ t = 1, 2, \dots, 5000 $).
$ X^{\{t\}} $ has dimensions $ n_x \times 1000 $ (number of features by number of examples in the mini-batch), and $ Y^{\{t\}} $ has dimensions $ 1 \times 1000 $ (for a single output per example).
This is different from:

$ X^{(i)} $: Refers to the $ i $-th training example.
$ W^{[l]} $: Refers to weights in layer $ l $ of the neural network.

![Alt Text](https://lh6.googleusercontent.com/9XzCnVwbAkrgxCfI6pN769ybYwYp--92xsrRXTXHx2FXGNYXf7rE_G7r6186lFp0_Pe6f09Yhncg7P0VHv59_4ZfrczgrvehudW0UQ0Ira7Y4VD6gROm_KU4leR4IeJcYiEFVbFj)
---
## The exponentially weighted average
- The exponentially weighted average for day $ t $ is:
$$V_t = \beta V_{t-1} + (1 - \beta) \theta_t$$

$ V_t $: The smoothed average at time $ t $.
$ \theta_t $: The actual value (e.g., temperature) at time $ t $.
$ \beta $: A number between 0 and 1 (e.g., 0.9) that controls how much weight goes to past averages vs. the current value.
$ V_{t-1} $: The previous day’s smoothed average.


- How It Smooths:
   If $ \beta = 0.9 $, then $ 1 - \beta = 0.1 $. This means 90% of $ V_t $ comes from the previous average ($ V_{t-1} $) and only 10% from the current temperature ($ \theta_t $).
   This reduces the impact of sudden changes (e.g., a very hot day), creating a smoother trend.


- Averaging Window:

1. The formula acts like it’s averaging over roughly $ \frac{1}{1 - \beta} $ days.
1. For $ \beta = 0.9 $, it’s about 10 days, meaning the average reflects temperatures from the last 10 days, with exponentially less
   weight for older days.
1. For $ \beta = 0.98 $, it’s about 50 days (smoother but slower to adapt).
1. For $ \beta = 0.5 $, it’s about 2 days (noisier but faster to adapt).


- Why Exponential?:

The weights decrease exponentially for older data. If you expand the formula:
$$V_t = (1 - \beta) \theta_t + \beta (1 - \beta) \theta_{t-1} + \beta^2 (1 - \beta) \theta_{t-2} + \dots$$
The weights ($ \beta^k (1 - \beta) $) get smaller for older days ($ \theta_{t-k} $), so recent data matters more.

 - In optimization algorithms, instead of averaging temperatures, you average gradients (changes in the loss function) to make updates smoother and training faster.
 - The smoothed gradients help the model avoid getting stuck or overreacting to noisy data.

---
## Bias Correction in Exponentially Weighted Averages

1. What It Is: A method to fix early inaccuracies in exponentially weighted averages (used in optimizers like Adam).
2. Why It’s Needed:
   When starting with a zero-initialized average, early estimates are too low (biased).
   Bias correction adjusts these estimates to be more accurate.

3. How It Works:
   Divide the moving average by a correction factor: $ V_{t,\text{corrected}} = \frac{V_t}{1 - \beta^t} $.
$ V_t $: Moving average at time $ t $.
$ \beta $: Weight for past values (e.g., 0.9).
$ \beta^t $: Decreases as $ t $ (time step) increases, reducing correction over time.

4. Why It Matters:
  Improves accuracy in early steps (e.g., first few updates in training).
  Less impact later as $ \beta^t \to 0 $, making the average naturally accurate.

5. Example: In neural network training, bias correction ensures early gradient updates are reliable, speeding up learning.
![Alt Text](https://gaussian37.github.io/assets/img/dl/dlai/bias_correction_exponentially_weighed_averages/1.png)

---
## RMSProp
 Gradient descent (method to update weights to minimize loss) can be slow and oscillate (wiggle up and down) in some directions, slowing progress.
 RMSProp reduces these oscillations, allowing faster training with a larger learning rate (step size for weight updates, denoted $ \alpha $).

- Problem with Gradient Descent:
  - Imagine a loss function shaped like a bowl, with a steep slope in one direction (e.g., vertical, called $ b $) and a gentle slope in another (e.g., horizontal, called $ W $).
  - Gradients in the vertical direction ($ \text{db} $) are large, causing big oscillations, while horizontal gradients ($ \text{dW} $) are smaller, leading to slow progress.
  - Goal: Slow down updates in the vertical direction (reduce oscillations) and speed up in the horizontal direction.

- How RMSProp Works:

For each mini-batch at iteration $ t $:

Compute gradients: $ \text{dW} $ (gradient for weights) and $ \text{db} $ (gradient for biases).
Keep an exponentially weighted average of squared gradients:
$$S_{\text{dW}} = \beta_2 S_{\text{dW}} + (1 - \beta_2) (\text{dW})^2$$
$$S_{\text{db}} = \beta_2 S_{\text{db}} + (1 - \beta_2) (\text{db})^2$$

$ (\text{dW})^2 $: Element-wise squaring (each gradient component is squared individually).
$ \beta_2 $: Hyperparameter (e.g., 0.9), similar to momentum’s beta but distinct to avoid confusion.


Update weights and biases:
$$W = W - \alpha \frac{\text{dW}}{\sqrt{S_{\text{dW}} + \epsilon}}$$
$$b = b - \alpha \frac{\text{db}}{\sqrt{S_{\text{db}} + \epsilon}}$$

$ \epsilon $: Small number (e.g., $ 10^{-8} $) added to avoid dividing by zero, ensuring numerical stability (preventing errors from very small denominators).

- Intuition:

Large gradients (e.g., $ \text{db} $ in vertical direction) make $ S_{\text{db}} $ large, so $ \sqrt{S_{\text{db}}} $ is large, reducing the update ($ \frac{\text{db}}{\sqrt{S_{\text{db}}}} $) to dampen oscillations.
Small gradients (e.g., $ \text{dW} $ in horizontal direction) make $ S_{\text{dW}} $ small, so $ \sqrt{S_{\text{dW}}} $ is small, keeping the update larger for faster progress.
Result: Smoother path toward the minimum, allowing a larger $ \alpha $ without diverging.

---

### **Gradient Descent with Momentum**

- **What It Is**: An optimization technique that speeds up gradient descent by averaging gradients.
- **How It Works**:
  - Compute an exponentially weighted average of past gradients (like a running average).
  - Use this average to update weights, instead of raw gradients.
- **Benefits**:
  - **Reduces Wiggles**: Smooths out zig-zags in the path to the minimum of the loss function.
  - **Faster Learning**: Allows a larger learning rate without overshooting, speeding up convergence.
- **Intuition**:
  - Imagine a ball rolling down a hill: Momentum remembers the direction, reducing side-to-side wobbles and moving faster toward the bottom.
  - For elongated loss functions, it slows updates in steep directions (less oscillation) and speeds up in flat directions (faster progress).
- **Example**: In training a CNN to recognize cats, momentum helps the model move steadily toward the best weights, avoiding erratic updates.

---

### **Adam Optimizer**

- **What It Is**: A powerful optimization method combining **Momentum** and **RMSprop** for fast, efficient neural network training.
- **How It Works**:
  - **Momentum Part**: Tracks an average of gradients to keep updates moving in the right direction, smoothing the path.
  - **RMSprop Part**: Adjusts step sizes based on gradient magnitude—smaller steps in steep areas, larger in flat areas.
  - Steps:
    1. Maintain two averages: one for gradients (\( V_t \)), one for squared gradients (\( S_t \)).
    2. Apply bias correction to both averages for early accuracy.
    3. Update weights using these corrected averages and the learning rate.
- **Why It’s Great**:
  - Adapts learning rate for each parameter, making training faster and more stable.
  - Combines smooth movement (Momentum) with smart step sizing (RMSprop).
- **Example**: For a self-driving car model, Adam quickly adjusts weights to detect road signs by balancing speed and stability.

---

### **Learning Rate Decay**

- **What It Is**: Gradually reducing the learning rate (\( \alpha \)) over training to fine-tune updates.
- **Why It’s Needed**:
  - Early on, a large learning rate speeds up learning.
  - Later, a smaller rate avoids overshooting the optimal weights.
- Common Methods:

Square Root Decay: $ \alpha = \frac{\text{constant}}{\sqrt{\text{epoch\_num}}} \cdot \alpha_0 $

Reduces $ \alpha $ based on the square root of the epoch number.


Exponential Decay: $ \alpha = \alpha_0 \cdot e^{-\text{decay\_rate} \cdot \text{epoch\_num}} $

Smoothly decreases $ \alpha $ exponentially.


Step Decay: $ \alpha = \alpha_0 \cdot \text{decay\_rate}^{\lfloor \text{epoch\_num} / \text{decay\_steps} \rfloor} $

Drops $ \alpha $ by a factor every few epochs.

- **Why It Matters**: Ensures fast learning early and precise adjustments later, improving accuracy.
- **Example**: In a face recognition model, learning rate decay starts with big steps to learn rough features, then smaller steps to fine-tune details.

---
