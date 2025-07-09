# Predictive Modelling for Climate Change Adaptation


![Project Title Image](project_title_image.png)

## Project Summary

This project evaluates the application of machine learning for weather prediction to support **ClimateWins**, a European non-profit organization dedicated to humanitarian responses to extreme weather events. As a resource-constrained non-profit, ClimateWins required a cost-effective, data-driven method to anticipate dangerous conditions and shift from a reactive to a predictive operational stance.

The core of this project is a proof-of-concept supervised learning model that classifies a given day's weather as "favourable" or "dangerous" in mainland Europe, using over a century of historical weather data.

---

## Table of Contents

- [Project Summary](#project-summary)
- [Key Findings & Outcome](#key-findings--outcome)
- [Methodology](#methodology)
- [Tools and Technologies](#tools-and-technologies)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
- [Recommendations](#recommendations)

---

## Key Findings & Outcome

The analysis confirmed a statistically significant warming trend across Europe since 1960 and successfully demonstrated the viability of machine learning for this predictive task.

- **Most Accurate Model:** A tuned **Gated Recurrent Unit (GRU) model** achieved the highest accuracy of **98.1%** on the complex multi-station prediction problem.
- **Key Predictive Stations:** A Random Forest model identified **Düsseldorf, Maastricht, and Basel** as the most influential "hub" stations, making them critical for data monitoring.
- **Critical Indicators:** The most influential variables for prediction were **atmospheric pressure** and **minimum temperature**.


*Exploratory data analysis confirmed a significant, long-term increase in both mean and maximum temperatures across Europe.*

---

## Methodology

1.  **Business Requirements Deconstruction:** The project began with an initial analysis of the project brief to define scope, objectives, stakeholders, and constraints.
2.  **Exploratory Data Analysis (EDA):** A statistical trend analysis was performed on historical weather data (1960-2021) from the European Climate Assessment & Data Set project. This validated the project's core premise by confirming a statistically significant warming trend.
3.  **Data Cleaning & Preparation:** The raw dataset, containing observations from 18 stations, was cleaned. This involved aligning stations between feature and target sets, handling structural gaps (missing observation types), and final imputation.
4.  **Feature Engineering & Scaling:** Features were scaled using `StandardScaler` and reshaped into a 3D tensor `(samples, timesteps, features)` for deep learning models.
5.  **Comparative Model Evaluation:** A comprehensive suite of machine learning models was trained and evaluated, including:
    - Random Forest (for baseline and feature importance)
    - Dense Neural Network
    - 1D Convolutional Neural Network (CNN)
    - Recurrent Neural Networks (Simple RNN, LSTM, and GRU)
6.  **Hyperparameter Optimization:** Keras Tuner (`Hyperband`) was used to find the optimal architecture and hyperparameters for the deep learning models.

---

## Tools and Technologies

- **Python 3.x**
- **Jupyter Notebook**
- **Core Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (`RandomForestClassifier`, `RandomizedSearchCV`)
- **Deep Learning:** TensorFlow, Keras (`Sequential`, `GRU`, `LSTM`, `CNN`), Keras-Tuner

---

## Project Structure

```text
├── 01_roject_management/	<- Outputs: maps, dashboard, presentations, etc.
├── 02_data				<- Data files (raw and processed)
├── 03_notebooks			<- Jupyter notebooks for each step of the analysis
	└── scripts/	    <- Reusable Python scripts (if any)
├── README.md				<- Project documentation (this file)
├── .gitignore				<- Git ignore rules
```

## How to Reproduce

1.  Clone this repository:
    ```bash
    git clone https://github.com/kleptsoved/climate-weather-prediction_v02
    ```
2.  (Optional) Set up a virtual environment.
3.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras keras-tuner
    ```
4.  The primary analysis and model training pipeline can be found in the `notebooks/ml_pipeline_multimode.ipynb` file. Note that the original data files are not included in this repository due to their size.

---

## Recommendations

Based on the analysis, the following strategic recommendations were provided to ClimateWins:

1.  **Operationalize the GRU Model:** Implement the high-accuracy GRU model as a proof-of-concept for a regional weather alert system.
2.  **Prioritize Sensor Data:** Focus investment on high-quality sensor data for the top 3 predictive stations (Düsseldorf, Maastricht, Basel), particularly for pressure and temperature monitors.
3.  **Develop a Nuanced Risk Score:** Evolve the binary "dangerous" label into a multi-class "Operational Risk Score" (e.g., Low, Medium, High) for more granular mission planning.
