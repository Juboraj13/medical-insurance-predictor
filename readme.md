# Medical Insurance Cost Forecaster

An end-to-end Machine Learning pipeline engineered to predict medical insurance costs using patient demographic and lifestyle data. This project benchmarks a baseline Linear Regression against an optimized Random Forest Regressor to provide clean, automated pricing risk calculations.

 Project Overview

Traditional insurance underwriting often relies on rigid pricing sheets or simple linear equations that miscalculate compound health risks. This project builds a reliable machine learning pipeline using **Scikit-Learn** that takes attributes like age, BMI, and smoking habits to calculate an optimized annual premium quote.

### Key Pipeline Elements:
* **Robust Preprocessing:** Uses automated `ColumnTransformer` to handle scaling and text encoding simultaneously, completely removing the risk of data leakage.
* **Algorithmic Progress:** Transitions cleanly from a baseline ordinary least squares regression to a robust multi-tree ensemble method.
* **Hyperparameter Tuning:** Implements a 5-fold cross-validated grid search (`GridSearchCV`) to fine-tune tree depth and estimator constraints.



 Repository Layout

* `insurance_cost_predictor.ipynb` — The complete step-by-step Jupyter Notebook containing the EDA, data visualizations, pipeline engineering, and evaluation phases.
* `requirements.txt` — Software package dependencies required to run this codebase locally.


Performance Metrics

The predictive frameworks were measured rigorously using Root Mean Squared Error (RMSE) and the Coefficient of Determination ($R^2$):

| Model Architecture | Error Margin (RMSE) | Variance Explained ($R^2$) |

| **Baseline Linear Regression** | \$5,796.28 | 78.36% |
| **Default Random Forest** | \$4,567.78 | 86.56% |
| **Tuned Random Forest (Optimal)** | **\$4,432.84** | **87.34%** |

### Core Discovery:
A straight line (linear modeling) fails to capture how risk compounding behaves in the real world. Our EDA showed that insurance charges take an aggressive, exponential turn specifically when a patient **both smokes and crosses a BMI of 30**. By optimizing a Random Forest to a max depth of 5 layers, our model effectively maps these non-linear conditions, cutting total calculation errors by **23.5%**.

---

 ##Setup & Installation

Follow these steps to run this project on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/medical-insurance-forecaster.git](https://github.com/YOUR_GITHUB_USERNAME/medical-insurance-forecaster.git)
cd medical-insurance-forecaster
