# Algorithmique: Uncertainty Quantification in Multi-Output Regression


This repository provides an R package implementing conformal prediction techniques to estimate simultaneous prediction intervals for multi-output regression tasks. The goal is to offer uncertainty-aware predictions with provable guarantees, even in high-dimensional settings.

We include:

Three uncertainty quantification methods (Beta-Optim, Max Rank, Fast Beta-Optim)

R + C++ implementations for comparison

A tutorial and benchmark results

## Problem & Motivation

In many fields such as health monitoring, energy prediction, or financial forecasting, we aim to predict not just a single value, but an entire curve (e.g., heart rate over 24 hours).

Yet, it's crucial to also answer:

🛡"How confident am I in these predictions?"

We want to provide prediction intervals that cover all dimensions of $Y = (y_1, y_2, \dots, y_p)$ simultaneously, with a global probability $1 - \alpha$.

To solve this, we use Conformal Prediction, and in particular, we propose:

Beta-Optim: a calibration-based method

Fast Beta-Optim: a rank-based acceleration

Max Rank: a simple, extremely fast baseline

## About Conformal Prediction

Conformal prediction is a statistical framework that allows us to construct prediction intervals that are valid **regardless of the underlying model**. It only requires that the data be **exchangeable**, which is a weaker condition than being i.i.d.

It provides the guarantee:  
P(Y_test ∈ C(X_test)) ≥ 1 − α  
for some significance level (α). This means that the true target will lie inside the predicted interval at least (1 − α) of the time.

![Conformal Prediction Illustration](figures/conformal_diagram.png)



**Coverage Guarantee**

**Exchangeability Assumption**: Assume that the calibration set \((X_i, Y_i)\) and the test point \((X_{\text{test}}, Y_{\text{test}})\) are exchangeable.

Then conformal prediction guarantees:

  **1 − α ≤ P(Y_test ∈ C(X_test)) ≤ 1 − α + 1 / (n + 1)**

This means we have a probabilistic bound on coverage even with finite calibration size.


## Features
- Generation of synthetic non-linear datasets with configurable output dimensions.
- Training of regression models using **polynomial regression** or **gradient boosting** (`xgboost`).
- Conformal prediction interval computation and coverage analysis.
- Tools for empirical evaluation and visualization.
- Fast C++ implementation for speed comparison with the R version.

## Installation
### R Package Dependencies
```r
install.packages(c("data.table", "xgboost", "caret", "R6", "Rcpp", "RcppArmadillo", "devtools", "roxygen2", "testthat"))
```

### Installing the Package from GitHub
If you organize the code as an R package:
```r
# devtools must be installed first
install.packages("devtools")
library(devtools)
install_github("your_username/Algorithmique")
```

### Setting Up a C++ Compiler (Windows Users)
Since Rcpp requires a C++ compiler, Windows users must install Rtools:
- Download and install from: https://cran.r-project.org/bin/windows/Rtools/

For general R updates and additional package downloads, visit: https://cran.r-project.org/

## Tutorial
A step-by-step tutorial is provided in `Tutorial.pdf`, which guides you through:
1. Generating a synthetic dataset
2. Splitting data into training, calibration, and test sets
3. Training a model with `MLModel`
4. Applying uncertainty quantification using the `ModelUncertainties` class
5. Visualizing prediction intervals

### Example Workflow
```r
# Train a model
model <- MLModel$new(X_train, y_train, method = "gradient_boosting")
model$fit()

# Initialize the uncertainty model
uncertainty_model <- ModelUncertainties$new(
  model = model,
  X_calibration = X_calib,
  Y_calibration = y_calib,
  uncertainty_method = "Beta_Optim",
  Global_alpha = 0.9
)

# Fit the uncertainty method (compute Beta quantiles)
uncertainty_model$fit()
```

## Implemented Methods
- **Beta-Optim:**
Binary search over $\beta$ to find the smallest value such that the simultaneous coverage is at least $1 - \alpha$.
  - This method searches for the smallest width of prediction intervals that ensures a target **simultaneous coverage**.
  - It uses **dichotomic optimization** on a parameter \( \beta \), which controls the tolerance of the intervals.
  - For each candidate \( \beta \), quantiles \( q_j(1-\beta) \) are computed per dimension, and coverage is evaluated.
$p$-dimensional quantiles for each $\beta$.
  - The optimal \( \beta^* \) minimizes the deviation from the desired coverage \( 1 - \alpha \).


```
simcov(β) = (1/n) * ∑_{i=1}^n ∏_{j=1}^p 𝟙{ y_{i,j} ∈ [ŷ_{i,j} ± q_j(1 - β)] }
```

🔹 Max Rank
- **Max Rank:**
  - Ranks residuals for each dimension, and uses the **maximum rank per individual** as a summary statistic.
  - Prediction intervals are constructed using a single quantile position (\( r_{\text{max}} \)) across dimensions.

- **Max Rank Beta-Optim (Fast Beta-Optim):**
  - Combines the **rank-based speed** of Max Rank with the **coverage optimization** of Beta-Optim.
  - Instead of optimizing each dimension’s quantile separately, we optimize directly on the rank threshold.
  - This results in a much faster algorithm with similar coverage properties.
 Fast Beta-Optim

simcov(beta) = (1/n) * sum over i=1 to n of:
               indicator{ R_max(i) <= ceil((1 - beta) * n) }

### Beta-Optim: Simultaneous Coverage

![Beta Optim Formula](/betaoptim.png)

This formula defines the coverage function that is minimized to find the optimal β.


## Results
- All three methods were tested on simulated datasets representing multivariate time series (e.g., biological or cardiac signals).
- Empirical evaluations show that Max Rank Beta-Optim provides the best trade-off between computational efficiency and predictive reliability.

## Perspectives
- Future work could involve extending the method to a vectorized beta, learning one \( \beta_j \) per output dimension.
- Integration with more complex deep learning models.
- Real-world applications on medical, industrial, or energy datasets.

## Project Context
This package was developed as part of a Master's project at Université Paris-Saclay, involving:
- Theoretical and practical study of **simultaneous conformal prediction**
- Implementation of three methods:
  - Beta-Optim (with dichotomic search)
  - Max Rank (rank-based thresholding)
  - Max Rank Beta-Optim (efficient coverage tuning)
- Evaluation on simulated multi-frequency data

Refer to `Rapport_Projet_Algorithmique.pdf` and `Algo_report_Aymane_Jaad_FatimaZ.pdf` for the complete methodology and results.

## License
MIT License

## Authors
- Jaad Belhouari
- Fatima-Zahra Hannou
- Aymane Mimoun

---

