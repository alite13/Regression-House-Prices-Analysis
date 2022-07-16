from data_preprocessing import InitialDataLoader
from models import ols, rmsle_cv, rmsle, AveragingModels, StackingAveragedModels, NNRegressor
from plots import ols_plots, ols_tests, rb_plots

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import sys
import pickle
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.iolib.smpickle import load_pickle
from sklearn.model_selection import train_test_split

# Ignore warnings
warnings.filterwarnings('ignore')
# To avoid warnings while running XGB regressor
xgb.set_config(verbosity = 0)


def submission(prediction, sub_name):

    """ Creates a .csv file with submission to kaggle """
    submission = pd.DataFrame({'Id': pd.read_csv('test.csv').Id, 'SalePrice': prediction})
    submission.to_csv('{}.csv'.format(sub_name), index = False)

if __name__ == '__main__':

    """ Runs training and testing of
    1) OLS Regression 
    2) Different sklearn robust regression models
    3) Neural Network for regression
    and makes final predictions to .csv file """

    models_folder = 'C:\\Users\\a_lite13\\Dropbox\\House-Prices\\models\\'
    initial_data_loader = InitialDataLoader('C:\\Users\\a_lite13\\Dropbox\\House-Prices\\train.csv', 'C:\\Users\\a_lite13\\Dropbox\\House-Prices\\test.csv')
    train_features = initial_data_loader.train_data_preprocessing()
    test_features = initial_data_loader.test_data_preprocessing()
    # Create pipelines for sklearn robust regression models
    lasso_model = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 3))
    enet_model = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005, l1_ratio = 0.9, random_state = 3))
    krr_model = KernelRidge(alpha = 0.6, kernel = 'polynomial', degree = 2, coef0 = 2.5)
    gboost_model = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05, max_depth = 4, max_features = 'sqrt',
                            min_samples_leaf = 15, min_samples_split = 10, loss = 'huber', random_state = 5)
    xgb_model = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma = 0.0468, learning_rate = 0.05, max_depth = 3, 
                        min_child_weight = 1.7817, n_estimators = 2200, reg_alpha = 0.4640, reg_lambda = 0.8571,
                        subsample = 0.5213, silent = 1, random_state = 7, nthread = -1)
    lgb_model = lgb.LGBMRegressor(objective = 'regression',num_leaves = 5, learning_rate = 0.05, n_estimators = 720,
                        max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319,
                        feature_fraction_seed = 9, bagging_seed = 9, min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)
    # Ensemble using averaging of elastic net, lgbm, xgb and lasso model
    averaged_models = AveragingModels(models = (enet_model, lgb_model, xgb_model, lasso_model))
    # Ensemble using stacking of elastic net, lgbm, xgb and lasso model as a meta model
    stacked_averaged_models = StackingAveragedModels(base_models = (enet_model, lgb_model, xgb_model), meta_model = lasso_model)
    #sys.exit()
    print('\n=========== OLS Regression Modeling  =============\n')
    # We run the OLS regressor 4 times each time with the updated training set based on the influential observations found in the previous set
    # Regressor 1
    ols_regr1, ols_rmsle = ols(train_features)
    print('RMSLE:', ols_rmsle)
    infl_obs = ols_plots(ols_regr1, train_features)
    # Regressor 2
    train_features2 = train_features.drop(infl_obs, axis = 0)
    train_features2 = train_features2.reset_index(drop = True)
    ols_regr2, ols_rmsle = ols(train_features2)
    print('RMSLE:', ols_rmsle)
    infl_obs2 = ols_plots(ols_regr2, train_features2)
    # Regressor 3
    train_features3 = train_features2.drop(infl_obs2, axis = 0)
    train_features3 = train_features3.reset_index(drop = True)
    ols_regr3, ols_rmsle = ols(train_features3)
    print('RMSLE:', ols_rmsle)
    infl_obs3 = ols_plots(ols_regr3, train_features3)
    # Regressor 4
    train_features4 = train_features3.drop(infl_obs3, axis = 0)
    train_features4 = train_features4.reset_index(drop = True)
    ols_regr4, ols_rmsle = ols(train_features4)
    print('RMSLE:', ols_rmsle)
    infl_obs4 = ols_plots(ols_regr4, train_features4)
    # Tests for regression assumptions
    sw_results, bp_results = ols_tests(ols_regr4, train_features4)
    print(sw_results, bp_results)
    # Save the latest OLS model
    name = 'ols_model.pickle'
    ols_regr4.save('C:\\Users\\a_lite13\\Dropbox\\House-Prices\\models\\' + name)
    print('\nOLS modeling has been completed. OLS model has been saved.\n')

    print('\n============= Making Prediction using OLS =============\n')
    loaded_ols = load_pickle(models_folder + name)
    model_coeffs = loaded_ols.params
    test_cols = model_coeffs.index
    test_cols = test_cols.tolist()
    del test_cols[0]
    test_features = test_features[test_cols]
    test_features = sm.add_constant(test_features)
    ols_result = loaded_ols.predict(test_features)
    # Create .csv file with predictions
    submission(ols_result, 'ols-submission')
    print('Predictions has been saved!')
    
    print('\n============= Robust Regression Modeling ==============\n')
    
    """ # Uncomment to use the dataset without influential observations removal as done above in OLS
    X = train_features.loc[:, train_features.columns != 'SalePrice']
    y = train_features['SalePrice']
    """
    # Train/test split of the training data with influential observations removed as done above in OLS
    X = train_features4.loc[:, train_features4.columns != 'SalePrice']
    y = train_features4['SalePrice']
    # We use 80% of the training data and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42, shuffle = True)
    # Lasso
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    cv_score = rmsle_cv(lasso_model, X_train, y_train)
    val_score = rmsle(y_test, lasso_pred)
    print('Lasso Model CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('Lasso Model Validation RMSE: {:.4f}'.format(val_score.mean()))
    # Elastic Net
    enet_model.fit(X_train, y_train)
    enet_pred = enet_model.predict(X_test)
    cv_score = rmsle_cv(enet_model, X_train, y_train)
    val_score = rmsle(y_test, enet_pred)
    print('\nE-Net Model CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('E-Net Model Validation RMSE: {:.4f}'.format(val_score.mean()))
    # Kernel Ridge Model
    krr_model.fit(X_train, y_train)
    krr_pred = krr_model.predict(X_test)
    cv_score = rmsle_cv(krr_model, X_train, y_train)
    val_score = rmsle(y_test, krr_pred)
    print('\nRidge Regression Model CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('Ridge Regression Model Validation RMSE: {:.4f}'.format(val_score.mean()))
    # Gradient Boosting
    gboost_model.fit(X_train, y_train)
    gboost_pred = gboost_model.predict(X_test)
    cv_score = rmsle_cv(gboost_model, X_train, y_train)
    val_score = rmsle(y_test, gboost_pred)
    print('\nGradient Boosting CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('Gradient Boosting Validation RMSE: {:.4f}'.format(val_score.mean()))
    # XGB
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    cv_score = rmsle_cv(xgb_model, X_train, y_train)
    val_score = rmsle(y_test, xgb_pred)
    print('\nXGB Model CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('XGB Model Validation RMSE: {:.4f}'.format(val_score.mean()))
    # LGBM Model
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    cv_score = rmsle_cv(lgb_model, X_train, y_train)
    val_score = rmsle(y_test, lgb_pred)
    print('\nLGB Model CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('LGB Model Validation RMSE: {:.4f}'.format(val_score.mean()))
    # Ensembles - Averaged models
    averaged_models.fit(X_train.values, y_train)
    av_pred = averaged_models.predict(X_test)
    cv_score = rmsle_cv(averaged_models, X_train, y_train)
    val_score = rmsle(y_test, av_pred)
    print('\nEnsembling (average) CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('Ensembling (average) Validation RMSE: {:.4f}'.format(val_score.mean()))
    # Ensembles - Stacked Models
    stacked_averaged_models.fit(X_train.values, y_train)
    sav_pred = stacked_averaged_models.predict(X_test)
    cv_score = rmsle_cv(stacked_averaged_models, X_train, y_train)
    val_score = rmsle(y_test, sav_pred)
    print('\nEnsembling (stacked average) CV RMSE: {:.4f} (Stdv {:.4f})'.format(cv_score.mean(), cv_score.std()))
    print('Ensembling (stacked average) Validation RMSE: {:.4f}'.format(val_score.mean()))
    # OLS predictions have 100% of the data while all sklearn predictions have 80% as we defined the data split like this
    # Thus, we choose the number of rows in ols predictions respectively for the best visualization experience
    ols_pred = ols_regr4.fittedvalues
    ols_pred = ols_pred.iloc[:len(lasso_pred)]
    # Plot of predictions of sklearn regression models 
    rb_plots(lasso_pred, enet_pred, krr_pred, ols_pred, gboost_pred, xgb_pred, lgb_pred, av_pred, sav_pred)
    # Save models
    model_name = models_folder + 'rb_model'
    model_name2 = models_folder + 'rb_model2'
    model_name3 = models_folder + 'rb_model3'
    pickle.dump(lasso_model, open(model_name, 'wb'))
    pickle.dump(lgb_model, open(model_name2, 'wb'))
    pickle.dump(xgb_model, open(model_name3, 'wb'))
    print('\nRobust Regression training has been completed.\n')

    print('\n======= Making Prediction using Robust Regression ========\n')
    # Choose only those features that were used in training
    test_features = test_features[X_train.columns]
    loaded_lasso = pickle.load(open(model_name, 'rb'))
    loaded_lgb = pickle.load(open(model_name2, 'rb'))
    loaded_xgb = pickle.load(open(model_name3, 'rb'))
    lasso_pred = loaded_lasso.predict(test_features)
    lgb_pred = loaded_lgb.predict(test_features.values)
    xgb_pred = loaded_xgb.predict(test_features.values) 
    # Create the ensemble of predictions         
    ensemble_model = lasso_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
    # Create .csv file with predictions
    submission(ensemble_model,'rb-submission')
    print('Predictions has been saved!')

    print('\n====== Neural Network Regression Modeling  =======\n')
    # We train the NN on the whole dataset using X and y we defined before doing sklearn regression
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.99, random_state = 42, shuffle = True)
    nn_regr = NNRegressor(X.shape[1])
    train_model = nn_regr.train(np.reshape(X, (-1, X.shape[1])), y)
    print('\nNeural Network training has been completed.\n')

    print('\n======= Making Prediction using Neural Network ========\n')
    model = nn_regr.load_model(train_model)
    predictions = model.predict(test_features)
    # Create .csv file with predictions
    submission(predictions[:,0],'nn-submission')
    print('Predictions has been saved!')