# Timing US Equity Factors with Machine Learning

This repository contains the implementation of the master’s thesis:

**"Timing US Equity Factors with Machine Learning: A Forecasting Approach Using Macroeconomic Predictors"**

The project explores the use of machine learning techniques to forecast US equity factor returns based on macroeconomic indicators. It replicates and extends the framework introduced in the AQR paper ["Can Machines Time Markets?"](https://www.aqr.com/Insights/Research/Journal-Article/Can-Machines-Time-Markets) by applying both linear and nonlinear models to time a broad set of factor strategies.

---

## Project Overview

- **Goal**: Forecast and time monthly equity factor returns using macroeconomic predictors
- **Models**:
  - Simple linear regression (matrix-based)
  - Ridge regression with Fourier-transformed features
- **Predictors**: Macroeconomic variables from Welch & Goyal (2008)
- **Target Variables**: Monthly excess returns from the JKP factor library and the HML factor
- **Metrics**:
  - Sharpe Ratio
  - Appraisal Ratio
  - Alpha T-statistic
  - Skewness

---

## 📊 Data Sources

- **Predictors:**  
  Macroeconomic variables are taken from the dataset published by *Welch & Goyal (2008)*, commonly used in return forecasting research. These variables include interest rates, inflation, dividend yields, valuation ratios, and other market indicators.

- **Target Variables (Factors):**  
  Monthly **excess returns** of U.S. equity factors are sourced from the **Jensen, Kelly, and Pedersen (JKP) factor library** (https://jkpfactors.com/), covering strategies such as value, momentum, size, quality, and others.  
  Additionally, the classic **HML (High Minus Low)** factor from the Fama-French data is included for direct comparison with prior research (e.g., AQR’s study).

---

## 🧪 Performance Evaluation

Each model's predictive ability is assessed using the following financial metrics:

- **Sharpe Ratio** — Risk-adjusted return of the timing strategy.
- **Appraisal Ratio** — Measures alpha relative to tracking error vs. the factor.
- **Alpha T-statistic** — Statistical significance of the strategy's excess return.
- **Skewness** — Captures the asymmetry in return distributions.
- **Factor Sharpe & Skew** — Also computed for the raw underlying factor returns for benchmarking.

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.


