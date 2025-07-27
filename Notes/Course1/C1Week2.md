# Binary Classification
In a binary classification problem the result is a discrete value output.
For example, If the image is cat then(1) and if not a cat then(0) where the value of y is used to denote the output label of either 1 or 0.

---

# Image Representation
An image is represented in form of 3 RGB 64 Matrices. 
If input image is 64*64 pixels then we will have 3 corresponding RGB matrices of 64 pixels.
The dimension of feature vector x is denoted by nx and is equal to 12288.

![Image Represent](Images/ImgRepresent.png)

---

# Notation for training examples
They are represented by (x,y) where x is the feature vector of dimesnion nx and y is binary output having values 0 and 1.

In similar manner m is used to denote the training examples and M can be put togehter to make test sets.

The feature vector in each training examples is put together as columns so they have m input feature vector from the training sets nd have nx number of rows.

`X ∈ ℝⁿˣ x m`: Is an input matrix where each column is an input feature vector with `nₓ` features, and there are `m` examples in total.  
`X.shape = (nₓ, m)`
`Y ∈ ℝ¹ × m`: Is a row vector containing all labels.  
`Y.shape = (1, m)`

![Notation](Images/Notation.png)


