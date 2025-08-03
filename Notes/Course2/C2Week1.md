## Train/Dev/Test Sets

- **Training Set** – Used to train the model and learn its parameters.
- **Development (Dev) Set** – Used to tune hyperparameters and compare models.
- **Test Set** – Used for final unbiased evaluation of selected model.

### Dataset Splits

- **Small Datasets** – (70% Train / 30% Test) or (60% Train / 20% Dev / 20% Test)
- **Large Datasets** – (98% Train / 1% Dev / 1% Test)

> Dev and Test sets should come from the same distribution as the training data.  
> If no Test set is available, the Dev set can act as both Dev and Test, but this increases the risk of overfitting.

---

#### Bias and Variance

- **Variance**: Measures how much the model’s predictions change when trained on different subsets of the data.
- **Bias**: Refers to the error introduced when a model makes strong assumptions and oversimplifies the data.

---
#### Train Set and Dev Set Error

- **Train Set Error**: The error measured on the data that the model was trained on. It tells us how well the model has learned the training data. 
- **Dev Set Error**: The error measured on a development set not seen during training. It helps to evaluate the model's ability to generalize to unseen data.  

#### Error Diagnosis Table

| Training Error | Dev Set Error | Diagnosis                        | Train Set Error % | Dev Set Error % |
|----------------|----------------|----------------------------------|--------------------|------------------|
| High           | High           | High bias (underfitting)         | 15%                | 16%              |
| Low            | High           | High variance (overfitting)      | 1%                 | 11%              |
| High           | Higher         | High bias and high variance      | 15%                | 30%              |
| Low            | Low            | Low bias and low variance (ideal)| 0.5%               | 1%               |

---

#### Bias-Variance Trade-off

It is a balance between a **simple model** (high bias, low variance) and a **complex model** (low bias, high variance) to achieve the best performance on **unseen new data**.

---

### Solutions

#### High Bias (Underfitting)
- Increasing the model complexity by adding more layers or units
- Training model for more epochs/iteration

####  High Variance (Overfitting)
- Collect more training data
- Using regularization (L2, dropout)

