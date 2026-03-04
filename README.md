**Electricity Demand Forecasting using Random Forest**

Project Overview

This project implements a machine learning pipeline to predict hourly electricity demand. By analyzing historical load data alongside solar and wind generation proxies, the model identifies daily and weekly consumption patterns to provide accurate forecasts.
Key Objectives:Predictive Modeling: 
1. Utilize Random Forest Regression to forecast grid load.
2. Feature Engineering: Resolve model extrapolation issues using time-series lag features.
3. Performance Evaluation: Validate model accuracy using R-squared ($R^2$), Mean Absolute Error (MAE), and RMSE.


**Technologies Used**
Language: Python

Libraries: * Pandas: Data manipulation and time-series alignment.
Scikit-Learn: Machine Learning (RandomForestRegressor).
Matplotlib: Static data visualization.
NumPy: Numerical computations.

**📂 Project Structure**
mL-load-prediction-project.py: Main execution script containing the pipeline.

demo.xlsx: Historical dataset (Hourly timestamps, Solar, Wind, Demand).

demand_forecast_graph.png: Output visualization.

//// The data used to train the model is given in the file named demo.xlsx, for your reference it is added in the repository. The data is processed and cleaned and they are real values obtained from 
Power Grid Corporation of India ////


**Future Improvements**
External Factors: Integrate Temperature and Humidity data via weather APIs.

Holiday Calendar: Incorporate government holidays to account for non-standard demand drops.

Advanced Architectures: Experiment with LSTM or XGBoost for potentially higher accuracy.

