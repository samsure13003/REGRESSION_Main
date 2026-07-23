# 🧠 Model Selection & Overfitting — Q&A

Notes on *why* this project compares multiple regression models instead of just using the best-scoring one, written up as a Q&A for anyone reviewing the repo (recruiters, collaborators, future me).

---

### Q1. Why use Ridge, Lasso, and ElasticNet at all if we already have Linear Regression?

Plain Linear Regression has one weakness: if features are correlated with each other (which is why this project applies a 0.85 correlation threshold to drop redundant features), the model can become unstable — small changes in data can swing coefficients wildly, and it tends to **overfit** on small datasets like this one (244 rows).

Ridge, Lasso, and ElasticNet all add a "penalty" that keeps the model's coefficients smaller and more controlled:

| Model | What it does | Analogy |
|---|---|---|
| **Ridge** | Shrinks all coefficients evenly | Puts every feature "on a diet" a little |
| **Lasso** | Shrinks some coefficients to exactly zero | Deletes useless features entirely |
| **ElasticNet** | Mix of both | Diet + occasional deletion |

They weren't used because Linear Regression was *bad* — it actually won on test R². They were used to **check whether regularization would help or hurt**, and to confirm Linear Regression's good score wasn't a fluke from overfitting. Running that comparison — instead of just picking the first model that worked — is itself the point.

---

### Q2. Why the "CV" versions — RidgeCV, LassoCV, ElasticNetCV?

Ridge, Lasso, and ElasticNet each have a hyperparameter called `alpha`, which controls *how strong* the penalty is. Pick it wrong, and the penalty is either too weak (no benefit) or too strong (underfits).

`RidgeCV`, `LassoCV`, and `ElasticNetCV` automatically test a range of `alpha` values using cross-validation (splitting the training data into folds, testing each alpha, and picking the best one) instead of relying on a manual guess.

- `Ridge()` → uses a guessed/default alpha (1.0)
- `RidgeCV()` → searches for the best alpha itself

It's a more rigorous, less arbitrary way of tuning the same model.

---

### Q3. The models gave different results — is that a problem?

No — it's expected, and it's the whole point of comparing them.

| Model | R² (test set) |
|---|---|
| Linear Regression | 98.5% |
| Ridge / RidgeCV | 98.4% |
| LassoCV / ElasticNetCV | 98.1% |
| Lasso | 94.9% |
| ElasticNet | 87.5% |

The plain `Ridge()`, `Lasso()`, and `ElasticNet()` (no CV) scored noticeably worse than their CV counterparts. That's because they use a **default alpha of 1.0**, which turned out to be too strong for this data — over-penalizing and **underfitting** (squashing useful coefficients too aggressively). The CV versions searched for a better alpha and recovered most of the performance.

**Takeaway:** default regularization strength hurt performance here; CV-tuned alpha recovered it. That's a useful, honest finding — not a failure.

---

### Q4. How do you actually know if a model is overfitting?

Test-set R² alone doesn't tell you that — you need to compare **train** performance against **test** performance:

```python
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Train R²: {train_score:.4f} | Test R²: {test_score:.4f}")
```

| Pattern | Meaning |
|---|---|
| Train ≈ Test (e.g. 98% vs 97%) | Good — the model generalizes well |
| Train >> Test (e.g. 99% vs 80%) | **Overfitting** — the model memorized training data |
| Both low | **Underfitting** — model too simple, or over-regularized |

With only 244 rows in this dataset, overfitting is a real risk worth checking for every model — not just the one with the best headline score.

---

### Q5. So which model should actually be chosen — isn't Linear Regression the best?

Ridge/Lasso/ElasticNet are not "overfitting detectors" — they're alternative models that could be deployed instead of Linear Regression. The train-vs-test comparison (Q4) is the actual overfitting detector, and it needs to be run on **every** model, including Linear Regression, before declaring a winner.

The decision process:

1. **Run train vs. test R² for all models first.** This reveals which models are overfitting.
2. **Compare only among the models that generalize well.**

Two possible outcomes:

- **If Linear Regression's train ≈ test score** → it isn't overfitting, and it's the right choice: simplest model, best score, no need for added regularization complexity.
- **If Linear Regression's train >> test score** → it's overfitting, and the 98.5% test R² is partly memorization. In that case, **RidgeCV** becomes the better pick — nearly identical test score (98.4%) but the penalty adds stability and makes it less likely to overfit on new, unseen data.

**Working assumption:** with only 244 rows, some overfitting is likely, and Linear Regression vs. RidgeCV are close enough (98.5% vs 98.4%) that RidgeCV is the safer "production" choice even if Linear Regression narrowly wins on paper — the small gap doesn't offset the stability regularization provides. This should be confirmed with the train/test check in Q4 rather than assumed.

---

## Summary

- Comparing multiple models isn't about finding the "best number" — it's about stress-testing whether the best number is trustworthy.
- Regularized models (Ridge/Lasso/ElasticNet) exist to control instability from correlated features, not to replace Linear Regression by default.
- CV variants remove manual guesswork from hyperparameter tuning.
- Different scores across models are informative, not a bug.
- Final model choice should be based on train-vs-test generalization, not test R² alone.
