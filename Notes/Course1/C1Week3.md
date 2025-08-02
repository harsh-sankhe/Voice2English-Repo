# Neural Network Representation

A neural network is a type of machine learning model inspired by the structure and function of the human brain. It consists of interconnected neurons (also called nodes) that are designed to recognize patterns and relationships in data.

It consists of three layers:

- **Input Layer** – Takes the input `X`, which can consist of multiple parameters such as `{x1, x2, x3, x4}`.
- **Hidden Layer** – Performs computation on the input `X` using interconnected neurons and parameters. It applies weights and biases to learn patterns in the data. There can be multiple hidden layers.
- **Output Layer** – Generates the final output `Y` based on the input and features learned in the hidden layer by neurons.

- The training set consists of input values `X` and output values `Y`, which are used to train the hidden layer parameters of the neural network.

![Neural Network Structure](images/NNStructure.png)

---

# Computing Neural Network Outputs

Each node in the hidden layer of a neural network performs two main computations.  
• First, it calculates the linear combination z using the input values along with the corresponding weights and biases.  
`z[l] = w[l] * a[l-1] + b[l]`  
• Second, it passes this z value through an activation function g() to produce the activated output a.  
`A[l] = activation(Z[l])`  
This output then becomes input for the next layer of the network.

• To make these computations faster and more efficient, especially when dealing with large datasets and multiple nodes and layers, we use vectorization instead of using explicit for-loops.  
• That means we handle entire layers at once by representing the weights, biases, inputs, and outputs as matrices.  
• When dealing with multiple neurons in a layer, we stack their parameters and outputs vertically in a matrix form and thus apply vectorization.

![Node Computation 1](images/Node.png)

For a two-layer neural network (1 hidden layer + 1 output layer), forward propagation is done as follows:

**Hidden Layer (Layer 1):**
- Stack all weights into a matrix: `W[1]`  
- Stack all biases into a vector: `b[1]`  
- Compute:  
  - `Z[1] = W[1] · X + b[1]` (Linear combination)  
  - `A[1] = σ(Z[1])` (Activation using sigmoid or ReLU)

**Output Layer (Layer 2):**
- Compute:  
  - `Z[2] = W[2] · A[1] + b[2]`  
  - `A[2] = ŷ = σ(Z[2])` (Final prediction)

![Node Computation 2](images/Representation.png)

In a neural network, the output computation can be seen as **iterative applications of logistic regression** at each layer, including the output layer.

---

# Vectorizing Across Multiple Examples

We majorly train our neural networks on multiple training examples at once (batch processing). 
To make this efficient, we vectorize the computations across all examples instead of processing one example at a time and using the loop structure

![Vectorization Across Examples](images/vectorizing.png)

---

# Activation Functions


# Random Initialization

•   Random weight initialization is essential as it helps to break symmetry between neurons. 
•   This ensures that neurons can learn different patterns; otherwise, they behave identically and end up learning the same thing.

It is safe and common to initialize all biases to zero.

Choosing the right scale for random weights is important:
- **Too small** → leads to slow learning (vanishing gradients).
- **Too large** → leads to unstable learning (exploding gradients).

**Proper initialization results in faster and more stable training.**
