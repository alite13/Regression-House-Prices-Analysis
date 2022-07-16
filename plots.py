import warnings
import numpy as np
from scipy import stats
from scipy.stats import shapiro
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.compat import lzip

warnings.filterwarnings('ignore') 
    
def ols_plots(regressor, df):

    """ OLS Regression Plots """
    
    X = df.loc[:, df.columns != 'SalePrice']
    y = df['SalePrice']
    residuals = regressor.resid
    fitted_values = regressor.fittedvalues
    stand_resids = regressor.resid_pearson
    influence = regressor.get_influence()
    leverage = influence.hat_matrix_diag
    
    lm_cooksd = influence.cooks_distance[0]
    critical_d = 4/len(X)
    print('Critical Cooks distance:', critical_d)
    out_d = lm_cooksd > critical_d
    infl_obs = X.index[out_d].tolist()
    print('Influential observations:\n', infl_obs)
    #print("Corresponding Cook's Distances:\n", lm_cooksd[out_d])
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    fig1, ax1 = plt.subplots()
    
    # Plot of influential observations
    sns.set(rc={"figure.figsize": (12, 8)})
    sm.graphics.influence_plot(regressor, criterion = 'cooks')
    
    # Plot of regression actual vs fitted
    ax1.set_title('Regression Actual vs Fitted Values Plot', fontsize = 22)
    ax1.scatter(y, fitted_values, c='crimson')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    p1 = max(max(fitted_values), max(y))
    p2 = min(min(fitted_values), min(y))
    ax1.plot([p1, p2], [p1, p2], 'b-')
    ax1.set_xlabel('True Values', fontsize = 16)
    ax1.set_ylabel('Predictions', fontsize = 16)
    ax1.axis('equal')
    fig1.savefig('./plots/ols-fitted-vals.png', facecolor = 'white', transparent = False)
    
    sns.set(rc={"figure.figsize": (20, 15)})
    # Plot of Residuals vs Fitted
    ax[0, 0].axhline(y = 0, color = 'grey', linestyle = 'dashed')
    ax[0, 0].set_xlabel('Fitted Values Plot')
    ax[0, 0].set_ylabel('Residuals')
    ax[0, 0].set_title('Residuals vs Fitted Fitted', fontsize = 15)
    sns.scatterplot(x = fitted_values, y = residuals, ax = ax[0, 0], color = 'mediumseagreen')
    
    # Normal Q-Q plot
    ax[0, 1].set_title('Normal Q-Q Plot', fontsize = 15)
    sm.qqplot(residuals, fit=True, line='45', ax = ax[0, 1])
    
    # Scale-Location Plot
    ax[1, 0].axhline(y=0, color='grey', linestyle='dashed')
    ax[1, 0].set_xlabel('Theoretical Percentiles')
    ax[1, 0].set_ylabel('Sample Percentiles')
    ax[1, 0].set_title('P-P Plot', fontsize = 15)
    probplot = sm.ProbPlot(residuals, stats.t, fit=True)
    fig = probplot.ppplot(line='45', ax=ax[1, 0])
    
    # Residual vs Leverage Plot
    ax[1, 1].axhline(y=0, color='grey', linestyle='dashed')
    ax[1, 1].set_xlabel('Leverage')
    ax[1, 1].set_ylabel('Sqrt(standardized residuals)')
    ax[1, 1].set_title('Residuals vs Leverage Plot', fontsize = 15)
    sns.scatterplot(x = leverage, y = stand_resids, ax=ax[1, 1])
    fig.savefig('./plots/ols-results.png', facecolor = 'white', transparent = False)
    #plt.tight_layout()
    #plt.show()
    return infl_obs

def ols_tests(regressor, X):
    
    """ OLS tests for regression assumptions """
    
    # Jarque-Bera test for normality of residuals ONLY FOR N > 2000m otherwise Shapiro Wilk
    #jb_names = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    #jb = sms.jarque_bera(regressor.resid)
    #jb_results = lzip(jb_names, jb)
    
    # Shapiro Wilk test
    sw_results = shapiro(np.sqrt(X))
    
    # Breush-Pagan test for heteroskedasticity
    bp_names = ["Lagrange multiplier statistic (BP test)", "p-value", "f-value", "f p-value"]
    bp = sms.het_breuschpagan(regressor.resid, regressor.model.exog)
    bp_results = lzip(bp_names, bp)

    # Harvey Collier test for linearity
    #hc_names = ['t-value', 'p-value']
    #hc = sms.linear_harvey_collier(regressor)
    #hc_results = lzip(hc_names, hc)
    
    return sw_results, bp_results

def rb_plots(pr1, pr2, pr3, pr4, pr5, pr6, pr7, pr8, pr9):
    plt.figure()
    plt.plot(pr1, 'gd', label='Lasso')
    plt.plot(pr2, 'b^', label='ElasticNet')
    plt.plot(pr3, 'ys', label='Ridge')
    plt.plot(pr4, 'x', label='OLSRegression')
    plt.plot(pr5, '*', label='GradientBoosting')
    plt.plot(pr6, '^', label='XGBRegressor')
    plt.plot(pr7, 'v', label='LGBMRegressor')
    plt.plot(pr8, 'o', label='AveragedModels')
    plt.plot(pr9, 'p', label='StackedModels')

    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
    plt.ylabel('Predicted', fontsize = 17)
    plt.xlabel('Training samples', fontsize = 17)
    plt.legend(loc="best")
    plt.title('Robust Regression Predictions and Their Average', fontsize = 25)
    plt.savefig('./plots/all-regression-results.png', facecolor = 'white', transparent = False)
    #plt.show()

