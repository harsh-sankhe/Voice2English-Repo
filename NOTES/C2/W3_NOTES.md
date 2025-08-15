# WEEK3

### **Hyperparameters in Neural Networks**

- **What Are They?**:
  - Settings you choose before training a neural network to control its structure and learning process.
  - Not learned from data, but adjusted to improve the model’s performance.
- **Examples**:
  - **Learning Rate ($ \alpha $)**: Most important; controls how fast the network updates weights.
  - **Momentum Term ($ \beta $)**: Smooths gradient updates for faster learning.
  - **Mini-Batch Size**: Number of examples processed per update.
  - **Regularization Parameters**: Prevent overfitting.
  - **Number of Layers/Units**: Defines network size.
  - **Activation Functions**: E.g., ReLU, sigmoid.
- **Why They Matter**:
  - Determine how weights ($ w $) and biases ($ b $) are learned, impacting model accuracy.
- **Example**: A high learning rate might make a CNN learn fast but miss the best weights, like speeding past a target.

---

### **Tuning Hyperparameters**

- **What It Is**: Testing different hyperparameter values to find the best combination for your model.
- **Methods**:
  - **Grid Search**: Test all combinations in a grid (e.g., $ \alpha = 0.1, 0.2 $, layers = 2, 3). Works for few parameters but slow.
  - **Random Sampling**: Pick random values (e.g., $ \alpha = 0.001, 0.5 $) to explore more possibilities faster.
  - **Coarse-to-Fine**: Start with random sampling over a wide range, then focus on a smaller, promising region with denser sampling.
- **Why It Works**: Random sampling tests diverse values, as it’s hard to know which hyperparameters matter most upfront.
- **Example**: For a face recognition CNN, random sampling might test $ \alpha = 0.01, 0.0001 $, finding 0.01 works best, then fine-tune around it.

---

### **Choosing the Right Scale for Hyperparameters**

- **Why Scale Matters**: Picking the right range for hyperparameters affects training speed and accuracy.
- **Types of Scaling**:
  - **Uniform Sampling**: Good for parameters like number of layers (e.g., 2 to 4) or units (e.g., 50 to 100).
  - **Logarithmic Sampling**: Better for wide-ranging parameters like learning rate ($ \alpha $, 0.0001 to 1) or momentum ($ \beta $, 0.9 to 0.999).
    - Formula: $ \alpha = 10^{R} $, where $ R $ is random between $-4$ and $0$ (e.g., $ \alpha $ from $ 10^{-4} $ to 1).
    - Ensures balanced exploration, especially for values near 1 where small changes matter.
- **Example**: For $ \alpha $, uniform sampling might miss small values (e.g., 0.0001), but logarithmic sampling evenly tests 0.0001 to 1, improving CNN performance.

---

### **Batch Normalization (BN)**

- **What It Is**: A technique to normalize layer inputs in a neural network for faster, more stable training.
- **Steps**:
  - **Compute Mean**: Average the activations ($ Z $) in a mini-batch.
  - **Compute Variance**: Measure how spread out the activations are.
  - **Normalize**: Subtract mean, divide by standard deviation (add small $ \epsilon $ for stability) to get zero mean, unit variance.
    Formula: $ \hat{Z} = \frac{Z - \mu}{\sqrt{\sigma^2 + \epsilon}} $
  - **Scale and Shift**: Scale and Shift: Multiply by learnable $ \gamma $, add learnable $ \beta $.
    Formula: $ Z_{\text{norm}} = \gamma \hat{Z} + \beta $
- **How It Fits in a CNN**:
  - Apply BN to activations before the activation function (e.g., ReLU).
  - Forward pass: Normalize at each layer for consistent inputs.
  - Backpropagation: Update weights $ W $, $\gamma$ , and $ \beta $ using gradients (bias b  is ignored as BN neutralizes it).
  - Optimize with algorithms like Adam.
- **BN During Testing**:
  - Training: Use mini-batch mean/variance.
  - Testing: Use running averages of mean/variance from training for single examples.
  - Deep learning frameworks handle this automatically.
- **Example**: In a CNN for cat images, BN ensures layer inputs stay consistent, speeding up training.

---

### **Why Batch Normalization Works**

- **Stabilizes Learning**: Normalizes inputs and hidden layer activations to similar scales, speeding up training.
- **Reduces Covariate Shift**: Keeps layer input distributions stable, so changes in early layers don’t disrupt later ones.
- **Improves Generalization**: Works better on new data (e.g., colored cats after training on black cats).
- **Slight Regularization**: Adds noise like dropout, reducing overfitting (less with larger mini-batches).
- **Example**: For a self-driving car CNN, BN helps the model train faster and handle varied road images.

---
## simpply ill say 
- **Hyperparameters**: “Think of them as knobs you tweak to make a neural network learn better, like adjusting the speed of a car.”
- **Tuning**: “Try random settings to find what works, like tasting different recipes, then zoom in on the best ones.”
- **Scaling**: “For some settings like learning rate, pick values on a log scale to test both tiny and big numbers evenly.”
- **Batch Normalization**: “It’s like keeping all ingredients in a recipe balanced so the cake bakes faster and better.”
- **Why BN Works**: “It smooths out training, like driving on a steady road, and helps the model work on new data.”

### **Softmax Regression**

- **What It Is**: A method to classify inputs into multiple classes (not just two, like logistic regression).
- **How It Works**:
  - **Output Layer**: Has $ C $ units (one per class, e.g., 4 for cat, dog, baby chick, none).
  - Computes a score ($ Z^{[L]} $) for each class: $ Z^{[L]} = W^{[L]} \cdot A^{[L-1]} + b^{[L]} $.
  - Applies **softmax activation** to turn scores into probabilities that sum to 1.
    - Step 1: Compute $ T = e^{Z^{[L]}} $ (element-wise exponentiation).
    - Step 2: Normalize: $ A^{[L]} = \frac{T}{\sum T} $ (e.g., $ A^{[L]}_i = \frac{e^{Z^{[L]}_i}}{\sum e^{Z^{[L]}_j}} $).
  - Output ($ \hat{Y} $): A vector of probabilities (e.g., [0.842, 0.042, 0.002, 0.114] for 4 classes).
- **Example**:
  - For $ Z^{[L]} = [5, 2, -1, 3] $:
    - $ T = [e^5, e^2, e^{-1}, e^3] \approx [148.4, 7.4, 0.4, 20.1] $.
    - Sum $ T = 176.3 $.
    - $ \hat{Y} = [148.4/176.3, 7.4/176.3, 0.4/176.3, 20.1/176.3] \approx [0.842, 0.042, 0.002, 0.114] $.
  - Highest probability (0.842) predicts class 0 (e.g., none).
- **Why It’s Useful**: Generalizes logistic regression for $ C \geq 2 $ classes, producing probabilities for each class.
- **Softmax vs. Hardmax**:
  - Softmax: Gives probabilities (e.g., [0.842, 0.042, 0.002, 0.114]).
  - Hardmax: Picks max score, outputs 1 for it, 0 for others (e.g., [1, 0, 0, 0]).
- **Example**: For an animal classifier (cat, dog, chick, none), softmax outputs probabilities to pick the most likely animal.

---

### **Training a Neural Network with Softmax**

- **Loss Function**:
  - For one example: $ L(\hat{Y}, Y) = -\sum_{j=1}^C Y_j \log(\hat{Y}_j) $.
  - If true class is $ j $, $ Y_j = 1 $, others 0, so loss simplifies to $ -\log(\hat{Y}_j) $.
  - Goal: Maximize probability of true class ($ \hat{Y}_j \approx 1 $).
  - Example: True class is cat ($ Y = [0, 1, 0, 0] $), predicted $ \hat{Y} = [0.3, 0.2, 0.1, 0.4] $.
    - Loss = $ -\log(0.2) \approx 1.609 $. Higher loss means poor prediction.
- **Cost Function**:
  - Average loss over all $ m $ training examples: $ J = \frac{1}{m} \sum_{i=1}^m L(\hat{Y}^{(i)}, Y^{(i)}) $.
  - Use gradient descent to minimize $ J $.
- **Backpropagation**:
  - Key derivative: $ dZ^{[L]} = \hat{Y} - Y $ (e.g., $ [0.3, 0.2, 0.1, 0.4] - [0, 1, 0, 0] = [0.3, -0.8, 0.1, 0.4] $).
  - Starts backpropagation to compute gradients for all layers.
  - Frameworks like TensorFlow handle this automatically if forward propagation is set up.
- **Vectorized Implementation**:
  - $ Y $: Matrix of true labels ($ C \times m $, e.g., 4x$ m $).
  - $ \hat{Y} $: Matrix of predictions ($ C \times m $).
- **Example**: In a dog classifier, softmax training adjusts weights to increase the probability of “dog” for dog images, minimizing loss.
![Alt Text](https://media.geeksforgeeks.org/wp-content/uploads/20240706012340/Softmax-Activation-Function.webp)
---

### **Local Optima in Deep Learning**

- **Old Concern**: Early deep learning worried about getting stuck in local optima (bad low points in the loss function).
- **New Understanding**:
  - In high-dimensional spaces (e.g., 20,000 parameters), local optima are rare.
  - Most zero-gradient points are **saddle points** (some directions go up, others down), not local minima.
  - Probability of all directions being minima is tiny (e.g., $ 2^{-20000} $).
- **Real Problem**: **Plateaus**—flat regions where gradients are near zero, slowing down gradient descent.
  - Takes long to move through plateaus before finding a steep path down.
- **Solution**: Advanced optimizers (e.g., Momentum, Adam) speed up movement through plateaus.
- **Why It Matters**: Understanding saddle points and plateaus helps choose better optimizers for faster training.
- **Example**: In a CNN for face recognition, Adam helps escape plateaus, finding good weights faster than basic gradient descent.

---

- **Softmax**: “It’s like picking the most likely animal from a photo, giving each a probability score that adds to 100%.”
- **Training**: “The network learns by checking how wrong its guesses are and tweaking to make the right class score higher.”
- **Local Optima**: “Training can get stuck on flat spots, like a ball rolling slowly on a table, but smart methods like Adam help it roll faster.”

