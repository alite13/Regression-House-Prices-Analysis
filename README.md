# Analyzing and Predicting Housing Prices (Advanced Regression)  
This is the repository for the source code of the housing prices prediction and analysis using advanced regression modeling project found on   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques  
My Jupyter Notebook on Kaggle is available here ...  

## Dependencies (Anaconda):  

Scikit-learn 1.1.0 (`conda install -c conda-forge scikit-learn`)  

Scipy 1.8.1 (`conda install -c conda-forge scipy`)  

Numpy 1.23.1 (`conda install -c conda-forge numpy`)  

Matplotlib 3.5.2 (`conda install -c conda-forge matplotlib`)  

Seaborn 0.11.2 (`conda install -c conda-forge seaborn`)  

Pandas 1.4.3 (`conda install -c conda-forge pandas`)  

Statsmodels 0.13.2 (`conda install -c conda-forge statsmodels`)  

XGBoost 1.6.1 (`conda install -c conda-forge xgboost`)  

Lightgbm 3.3.2 (`conda install -c conda-forge lightgbm`)  

Tensorflow 2.3.0 (`conda install -c conda-forge tensorflow`)  

Keras 2.4.3 (`conda install -c conda-forge keras`)  

Joblib 1.1.0 (`conda install -c anaconda joblib`)  

This project is implemented using Python 3.8.5.

## Results:  

Model - the name of the model.  
Correlation threshold - the lowest allowed correlation value between the target and predictor in the training dataset.  
Input features - the number of input features to a model.  
Influential Points - calculated points that influence the fitted values using Cook's distance.  
Test RMSE - RMSE of the submitted predictions to Kaggle. 

| Model | Correlation Threshold | Input Features | Influential Points | Test RMSE | 
| --- | --- | --- | --- | --- |
| Ordinary Least Squares Regressor | 0 | 77 | Removed | 0.15499 |
| NN Regressor (Mean Absolute Error Loss) | 0 | 77 | Removed | 0.16985 
| Ensemble (Lasso + XGBoost + LGBM) Regressor | 0 | 77 | Removed |  0.13279 | 

Some intermediate experiments were also done. Those included feature engineering (new features, polynomial features, interaction terms), feature transformations, removal of outliers, and various correlations thresholds.
