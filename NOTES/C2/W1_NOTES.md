# COURSE 2
## WEEK 1
## Train / Dev / Test Sets

Split your data into:

- **Train**: for model training  
- **Dev (Development)**: for tuning model choices  
- **Test**: for final evaluation  

---

### Traditional Splits (Small Datasets)

- 70% Train, 30% Test  
- 60% Train, 20% Dev, 20% Test  

---

### Modern Splits (Big Data)

For **1 million examples**:

- 98% Train  
- 1% Dev (10,000 examples)  
- 1% Test (10,000 examples)  

### Example

- **Training**: Cat pictures from the web (high-res, professional)  
- **Dev/Test**: Cat pictures from users (blurry, casual)  

## Bias & Variance 
### The "True" Curve (Ideal Model)

There’s a real (but unknown) curve that perfectly connects **weight and height**.  
Since we don’t know it, we use **machine learning** to approximate this relationship.

---

## Training vs Testing Data

We split our data into two parts:

- **Training Set** – teaches the model  
- **Testing Set** – checks how well the model works on **new** data

---

### Bias = Too Simple

- A straight line model (like **linear regression**) is **too simple** to capture the real pattern.  
- It **misses the true relationship**, even with lots of training.  
-  This is called **high bias** – the model is wrong because it’s too basic.

---

### Variance = Too curved

- A very flexible model (like a **curved line**) fits the training data **too well**.  
- But it performs **badly** on new data (testing set) because it's too sensitive to noise.  
- This is called **high variance** – the model changes too much when the data changes.

---

###  Overfitting

When a model does well on training data but badly on testing data, it's called **overfitting**.  
Like **memorizing answers** instead of actually understanding the topic.

---

### Finding Balance

The best models:

- Are **not too simple** (→ low bias)  
- Are **not too wiggly** (→ low variance)  
- They sit in the **"sweet spot"** between underfitting and overfitting.

![Alt Text](https://docs.aws.amazon.com/images/machine-learning/latest/dg/images/mlconcepts_image5.png) 

underfitting shows us high biasing 
overfitting = high variance 
now to find Use techniques like:

## Regularization
Regularization is a way to prevent overfitting.
When a neural network fits the training data too well, it might memorize noise and perform badly on new examples. Regularization adds a penalty to the cost function to keep weights small and discourage overly complex models.
Regularization works by adding a penalty to the cost function that grows when weights become large. Large weights allow the model to twist and bend to fit every little noise in the training data i.e. overfit. By penalizing large weights, regularization encourages the model to learn simpler, smoother mappings that capture the underlying trend but ignore noise.
- and the second term in l2 that had w^2 is sum of squared weights (Frobenius norm) across all layers, scaled by the regularization parameter λ


![Alt Text](https://media.licdn.com/dms/image/v2/D4D22AQFgccwyk6tBDQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1719410596016?e=2147483647&v=beta&t=-84x3E0W--aeREyu_KFblvSjmLXxLAbRVLS-p0wKanM)
![Alt Text](https://towardsdatascience.com/wp-content/uploads/2023/12/1xfOhMtPoJLuUiiaSovoDxA.png)
When we differentiate J with respect to W
- First term: comes from standard backpropagation (loss gradient without regularization).
- Second term: comes from differentiating the L2 penalty.
- Each update shrinks the weights by a factor (1 - αλ/m ) before applying the normal gradient update.
- This shrinkage is what’s called weight decay.
---
## How does Regularization prevent overfitting???

 High bias (left): Model is too simple (underfitting) → straight line, doesn’t capture curvature of data.
 Just right (middle): Good balance between bias and variance → decision boundary fits data well.
 High variance (right): Model is too complex (overfitting) → boundary twists around noise points.
- so accordingly note W-->0 shows that regularization encourages smaller weights.
Smaller weights mean the model’s complexity reduces, avoiding wild oscillations (like in high variance).
- When λ is larger, the penalty on large weights is bigger → network will shrink many weights toward zero → effectively simplifying the model and preventing overfitting.
if W is small then Z is small For tanh activation small z values are in the linear region (near 0).
When activations are in the linear region, the network behaves almost like a linear model, which is less likely to overfit.
- If λ is too small → regularization effect is negligible → risk of high variance (overfitting).
If λ is too large → weights shrink too much → underfitting (high bias).
The goal is finding λ that gives “just right” complexity.
---

## Dropout Regularization

- **Dropout** is a technique where we randomly turn off neurons during training — on purpose!
- At each training step, some neurons are temporarily "ignored" (i.e., set to `0`).
- During testing, we use the **full network** (all neurons active) but scale activations to match training behavior.

---

### Inverted Dropout

- Normally, if you drop neurons, the **total activation** of a layer decreases.
- **Inverted dropout** compensates for this during training by **scaling up** the remaining activations by:scale_factor = 1 / keep_prob

- **Why it’s good**:
- At test time, you don’t need to rescale anything — you just use the raw activations.
- Keeps the expected value of activations consistent between training and testing.

```
D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob

(inverted dropout scaling)
 A *= D          # Shut off some neurons
 A /= keep_prob  # Scale up others to balance output
 At Test Time

```

We don’t apply dropout during inference.
Instead, we use the full network, and the earlier scaling by keep_prob ensures consistency.


## Other Regularization Methods

### 1. Data Augmentation
- **Goal**: Increase the size of the training dataset without collecting new data.
- **Examples**:
  - Horizontally flipping images
  - Randomly cropping images
  - Applying small rotations and distortions
- **Benefit**:
  - Helps reduce overfitting by teaching the model that small changes to the input still represent the same class.
  - Improves generalization without requiring new, independent data.

---

### 2. Early Stopping
- **Process**:
  - Monitor the **validation error** during training.
  - Stop training when validation error **starts to increase** (sign of overfitting).
- **Benefit**:
  - Prevents overfitting by halting training before the model memorizes the training data.
  - Ensures good performance on **unseen data**.

---
## NORMALIZING INPUTS 
1. When you feed raw data to a neural network, each feature might have different scales.
Example:
- Height in cm → ~150–200
- weight in kg → ~50–90
2. Gradient descent works much faster if all features are on a similar scale.
3. How to normalize 
- Subtract the mean (μ) of each feature → shifts data so the mean is 0.
- Divide by the standard deviation (σ) → scales data so variance is 1.
- formula : **x_norm = σ/x−μ**
4. Without normalization, the cost function looks like a stretched ellipse (gradient descent zigzags → slow).
   - With normalization, it’s more circular → gradient descent heads straight to the minimum.

![Alt Text](https://towardsdatascience.com/wp-content/uploads/2021/09/1onZIiGguLfbUYs3aTtmijg.jpeg)

---
## VANISHING GRADIENTS
1. In deep networks, as you pass gradients backward layer-by-layer:
   - If weights are slightly > 1, gradients grow exponentially (exploding gradient).
   - If weights are slightly < 1, gradients shrink exponentially (vanishing gradient).
2. Effects: Exploding → gradients too large → unstable training.
   - Vanishing → gradients too small → almost no learning.
- This often happens if you don’t initialize weights properly.

---

## WEIGHT INITIALIZATION 
1. In neural networks, weights are parameters that determine the strength of connections between neurons. When training a neural network, these weights are initially set to random values. The choice of these initial values significantly impacts how effectively the network learns. Poor initialization can lead to vanishing gradients (gradients become too small, slowing or halting learning) or exploding gradients (gradients become too large, causing unstable training).
2. why is it important 
   - In deep networks, gradients are computed using the chain rule across many layers. If weights are too large, the gradients can grow exponentially (exploding gradients). If too small, they can shrink exponentially (vanishing gradients).
   - Proper initialization ensures that the activations and gradients remain within a stable range, enabling faster and more reliable training.
3. Single Neuron Example: For a neuron with inputs $ x_1, x_2, \dots, x_n $ and weights $ w_1, w_2, \dots, w_n $, the output before activation is:
    $$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$
   If $ n $ (number of inputs) is large, the weights $ w_i $ should be smaller to prevent $ z $ from becoming too large or too small.
4. Variance Scaling: To control the scale of $ z $, the weights are initialized with a variance that depends on the number of input features. For a general activation function, set:
$$\text{Var}(w_i) = \frac{1}{n}$$
where $ n $ is the number of input features to the neuron.
5. ReLU-Specific Initialization: For the ReLU (Rectified Linear Unit) activation function ($ g(z) = \max(0, z) $), a slightly modified variance is used:
$$\text{Var}(w_i) = \frac{2}{n}$$
This is known as He initialization, which accounts for the fact that ReLU discards negative values, effectively halving the variance of the output.

6. Tanh-Specific Initialization: For the tanh activation function, the variance is:
$$\text{Var}(w_i) = \frac{1}{n}$$
This is called Xavier initialization, suitable for activation functions like tanh or sigmoid that are symmetric around zero.

7. Implementation: In practice, weights are drawn from a Gaussian distribution (mean 0) and scaled by the square root of the desired variance. For a layer with $ n^{l-1} $ inputs (from the previous layer), the weights are initialized as:
$$w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n^{l-1}}}\right) \quad \text{(for ReLU)}$$
or
$$w \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n^{l-1}}}\right) \quad \text{(for tanh)}$$

8. Hyperparameter Tuning: The variance scaling factor (e.g., 1 or 2) can be treated as a hyperparameter and tuned, but this is typically a lower-priority hyperparameter compared to learning rate or network architecture.

## GRADIENT CHECKING
Gradient checking is a debugging tool used to ensure that the gradients computed via backpropagation in a neural network are correct. Backpropagation involves complex mathematical computations to calculate the gradient of the loss function with respect to the model’s parameters (weights and biases). Errors in implementation (e.g., incorrect derivatives or indexing) can lead to incorrect gradients, causing the model to learn improperly. Gradient checking approximates the gradient numerically and compares it to the analytically computed gradient from backpropagation to detect potential bugs.


- Backpropagation involves many computations (chain rule across layers), and small errors in the code can lead to incorrect gradients, which may go unnoticed but degrade model performance.
- Gradient checking provides a way to verify that the implemented gradients match the true gradients, increasing confidence in the correctness of the backpropagation code.
- It’s particularly useful during development to catch bugs early, ensuring the neural network trains as expected.

1. Numerical Gradient Approximation:

To check if the gradient of a function $ f(\theta) $ (e.g., the loss function) with respect to a parameter $ \theta $ is correct, we can approximate the derivative numerically using the two-sided difference method.
Instead of computing the gradient analytically (via backpropagation), we evaluate the function at $ \theta + \epsilon $ and $ \theta - \epsilon $, where $ \epsilon $ is a small value (e.g., 0.01).
The two-sided difference formula is:
$$\text{Approximated Gradient} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$
This approximates the derivative $ \frac{df}{d\theta} $ at $ \theta $.


2. Why Two-Sided Difference?

two-sided difference with the one-sided difference:
$$\text{One-Sided Difference} = \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$$

The two-sided difference is more accurate because it considers both sides of $ \theta $, forming a larger triangle (as visualized in the video) that better approximates the true slope of the function.
The error in the two-sided difference is on the order of $ \epsilon^2 $ (e.g., if $ \epsilon = 0.01 $, error ≈ $ 0.0001 $), whereas the one-sided difference has an error of order $ \epsilon $ (e.g., error ≈ $ 0.01 $), making the two-sided method much more precise.

3. Visual Intuition:
 the analogy of a triangle to explain the approximation. For a function $ f(\theta) $, the gradient is the slope at $ \theta $. The two-sided difference forms a larger triangle with base $ 2\epsilon $ (from $ \theta - \epsilon $ to $ \theta + \epsilon $) and height $ f(\theta + \epsilon) - f(\theta - \epsilon) $.
The slope of this triangle, $ \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} $, closely approximates the true derivative at $ \theta $.

4. Accuracy of Approximation:
The two-sided difference has an approximation error of order $ \epsilon^2 $, meaning the error scales with $ \epsilon^2 $ (e.g., if $ \epsilon = 0.01 $, error ≈ $ 0.0001 $).
The one-sided difference has a larger error of order $ \epsilon $ (e.g., error ≈ $ 0.01 $), making it less reliable.
This higher accuracy makes the two-sided difference the preferred method for gradient checking.

5. Application to Neural Networks:
In a neural network, $ \mathcal{L} $ is the loss function, and $ \theta $ represents all parameters (weights and biases).
Gradient checking computes the numerical gradient for each parameter and compares it to the backpropagation gradient. If the difference is small (e.g., less than $ 10^{-7} $), the implementation is likely correct.
The process is computationally expensive, as it requires evaluating the loss function twice per parameter, so it’s used only for debugging, not during training.

### practical tips for gradient checking to verify backpropagation:

1. Use it only for debugging, not training, because it’s slow.
2. Check individual gradient components to find bugs.
3. Include regularization in the loss function.
4. Turn off dropout during gradient checking.
5. Test gradients at initialization and after some training.