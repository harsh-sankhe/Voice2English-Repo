# Train/ Dev/ Test Sets

### Training Set
- Used to train the model.
- Model learns the parameters (weights and biases) from this data.

### Dev Set 
- Used to tune hyperparameters and select the best model.
- Helps evaluate how well the model generalizes to unseen data.

### Test Set
- Used only once after training and tuning.
- Evaluates the final performance of the model.

### Datasets Split Ratio
- The dataset is split into 98% training, 1% development , and 1% test set when we have large datasets.If we have a small dataset the general 60/20/20 split also works as well.
- It is important to make sure that the training set and the dev/test sets are all from the same distribution. For example - If the model is trained on high resolution images but the test set contains low resolution images the model will not work properly.
- It is ok in some cases to not have a test set and only a dev set.

---

# Bias and Variance

### Bias
- Error due to **over-simplified assumptions** in the model.
- High bias -> Model is too simple and misses relevant patterns (underfitting).
- Example: Predicting a curve with a straight line.

### Variance
- Error due to **model sensitivity to small fluctuations** in training data.
- High variance -> Model is too complex and fits noise in the training data (overfitting).
- Example: A very wiggly curve that perfectly fits training points but generalizes poorly.

### Bias-Variance Trade-off
| Scenario          | Bias      | Variance | Model Behavior             |
|-------------------|-----------|----------|----------------------------|
| Underfitting      | High      | Low      | Poor on train & dev        |
| Overfitting       | Low       | High     | Good on train, poor on dev |
| Just Right        | Low       | Low      | Good on train & dev        |

---



