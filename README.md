# Algorithmique: Uncertainty Quantification in Multi-Output Regression

## Overview
`Algorithmique` is an R package for building machine learning models and quantifying prediction uncertainty in multi-output regression problems using conformal prediction techniques. It provides multiple uncertainty estimation methods, including:
- **Beta-Optim**
- **Max Rank**
- **Max Rank Beta-Optim** (Fast Beta-Optim)

These methods enable you to construct **simultaneous prediction intervals** with valid statistical guarantees. The implementation also includes a comparison with a C++ version of the code to evaluate execution speed.

## Introduction & Motivation
In many real-world applications such as healthcare or energy monitoring, we aim to predict entire curves or trajectories (e.g., heart rate over 24 hours). These are inherently multi-dimensional outputs. While machine learning models can provide accurate point predictions, they often lack a measure of confidence. 

This project addresses the challenge of providing reliable **prediction intervals** that account for **uncertainty across all dimensions** simultaneously. The goal is to guarantee that the entire output vector lies within the predicted intervals with high probability.

To solve this, we adopt conformal prediction techniques, and in particular, we propose and analyze the performance of:
- **Beta-Optim**: A method to calibrate intervals by optimizing the tolerance parameter \( \beta \) through dichotomic search.
- **Max Rank**: A faster alternative that uses the maximum rank of residuals per individual to set interval widths.
- **Fast Beta-Optim (Max Rank Beta-Optim)**: A hybrid method combining the optimization of Beta-Optim with the speed and stability of Max Rank.

## About Conformal Prediction
Conformal prediction is a statistical framework that allows us to construct prediction intervals that are valid 
**regardless of the underlying model**. It only requires that the data be **exchangeable**, which is a weaker condition than being i.i.d.

It provides the guarantee:
\[
\mathbb{P}(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})) \geq 1 - \alpha
\]
for some significance level \( \alpha \). This means that the true target will lie inside the predicted interval at least \(1 - \alpha\) of the time.

![Conformal Prediction Illustration]

### Coverage Guarantee
**Exchangeability Assumption:** Assume that the calibration set \((X_i, Y_i)\) and the test point \((X_{\text{test}}, Y_{\text{test}})\) are exchangeable.

Then conformal prediction guarantees:
\[
1 - \alpha \leq \mathbb{P}(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})) \leq 1 - \alpha + \frac{1}{n + 1}
\]

This means we have a probabilistic bound on coverage even with finite calibration size \(n\).

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
  - This method searches for the smallest width of prediction intervals that ensures a target **simultaneous coverage**.
  - It uses **dichotomic optimization** on a parameter \( \beta \), which controls the tolerance of the intervals.
  - For each candidate \( \beta \), quantiles \( q_j(1-\beta) \) are computed per dimension, and coverage is evaluated.
  - The optimal \( \beta^* \) minimizes the deviation from the desired coverage \( 1 - \alpha \).

- **Max Rank:**
  - Ranks residuals for each dimension, and uses the **maximum rank per individual** as a summary statistic.
  - Prediction intervals are constructed using a single quantile position (\( r_{\text{max}} \)) across dimensions.

- **Max Rank Beta-Optim (Fast Beta-Optim):**
  - Combines the **rank-based speed** of Max Rank with the **coverage optimization** of Beta-Optim.
  - Instead of optimizing each dimension’s quantile separately, we optimize directly on the rank threshold.
  - This results in a much faster algorithm with similar coverage properties.

## Results
- All three methods were tested on simulated datasets representing multivariate time series (e.g., biological or cardiac signals).
- Empirical evaluations show that Max Rank Beta-Optim provides the best trade-off between computational efficiency and predictive reliability.
- Comparisons with the C++ implementation demonstrate significant speed-ups in execution.

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

