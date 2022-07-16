from re import L
import warnings
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class DfLabelEncoder():
    
    """ Class represents a multi column label encoder of a dataframe """
    
    def __init__(self, columns = None):
        self.columns = columns

    def fit(self,X,y = None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y = None):
        return self.fit(X,y).transform(X)

class InitialDataLoader():
    
    """ Class represents a data loader that performs data loading from csv files, 
        all preprocessing, and visualizations """
    
    def __init__(self, train_csv_dir, test_csv_dir):
        self.train_csv_dir = train_csv_dir
        self.test_csv_dir = test_csv_dir

    def missing_vals_plots(self, nans_num, nans_cat):

        """ Generates barplots of all missing values """

        sns.set(rc={'figure.figsize':(23, 26)})
        fig, ax = plt.subplots(nrows = 2, ncols = 1)
        ax[0].set_ylabel('Percent of NaNs', fontsize = 16)
        ax[0].set_title('NaN values in numerical features', fontsize = 24)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 65, fontsize = 14)
        sns.barplot(x = list(nans_num['col_name']), y = list(nans_num['percent_missing']), ax = ax[0])
        ax[1].set_ylabel('Percent of NaNs', fontsize = 16)
        ax[1].set_title('NaN values in categorical features', fontsize = 24)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 65, fontsize = 14)
        sns.barplot(x = list(nans_cat['col_name']), y = list(nans_cat['percent_missing']), ax = ax[1])
        fig.savefig('./plots/missing-vals.png', facecolor = 'white', transparent = False)
        
    def scatter_plots(self, df, order):
        
        """ Generates scatter plots """

        sns.set(rc={'figure.figsize':(90, 40)})
        cols = 12
        rows = round(df.shape[1]/cols)
        fig, ax = plt.subplots(rows, cols)
        fig.suptitle('Scatter Plots of Features', fontsize = 50)
        fig.subplots_adjust(top = 0.96)
        for var, subplot in zip(df, ax.flatten()):
            p = sns.regplot(x = var, y = 'SalePrice', data = df, ax = subplot, color = 'limegreen', line_kws={'color': 'b'}, order = order)
            p.set(ylabel = None)
            p.set_xlabel(var, fontsize = 25)
        fig.savefig('./plots/scatterplots.png', facecolor = 'white', transparent = False)
        #plt.show()

    def box_plots(self, df):
        
        """ Generates box plots to assess outliers """

        sns.set(rc={'figure.figsize':(50, 40)})
        cols = 12
        rows = round(df.shape[1]/cols)
        fig, ax = plt.subplots(rows, cols)
        fig.suptitle('Box Plots of Features', fontsize = 40)
        fig.subplots_adjust(top = 0.96)
        for var, subplot in zip(df, ax.flatten()):
            p = sns.boxplot(data = df[var], color = 'orchid', ax = subplot)
            p.set_xlabel(var, fontsize = 18)
        fig.savefig('./plots/boxplots.png', facecolor = 'white', transparent = False)
        #plt.show()
    
    def correlations(self, df, y, treshold):
        
        """ Calculates all correlations and generates a correlation matrix """

        pd.options.display.float_format = "{:,.2f}".format
        corr_matrix = df.corr(method = 'pearson')
        corr_matrix = corr_matrix.unstack()
        corr_matrix = corr_matrix[abs(corr_matrix) >= treshold]
        corr_col_names = corr_matrix['SalePrice'].index
        corr_features = df.loc[:, corr_col_names]
        train_features = pd.concat([corr_features, y], axis = 1)
        train_features = train_features.loc[:,~train_features.columns.duplicated()].copy()
        # Visualization
        corr_matrix = train_features.corr(method = 'pearson')
        sns.set(rc={'figure.figsize':(25, 21)})
        plt.title('Correlation Matrix', fontsize = 18)
        sns.heatmap(corr_matrix, mask = np.triu(np.ones_like(corr_matrix, dtype=bool)), vmax = 1.0, vmin = -1.0, linewidths = 0.1, annot_kws={"size": 9, "color": "black"}, square = True, cmap = 'PuOr', annot = True)
        return train_features
    
    def outlier_removal(self, df):
        
        """ Removes outliers based on predefined settings """

        df1 = df[~((df['Ground_SF'] > 7000))] 
        print('<<<', len(df) - len(df1), 'outliers have been removed >>> ')
        return df1
    
    def IQR_outlier_removal(self, df):
        
        """ Removes outliers based on IQR (not recommended in case of this project) """

        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df1 = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]
        df1 = df1.dropna()
        return df1
        
    """
    def histogram_plots(self, df_old, df_new):
        df_old_cols = df_old.columns.tolist()
        counts = list(range(0, len(df_old_cols)))
        sns.set(rc={"figure.figsize": (15, 180)})
        #fig1, axes1 = plt.subplots(rows, cols)
        fig, axes = plt.subplots(len(counts), 2)
        fig.subplots_adjust(top = 0.98)
        plt.suptitle('Histograms of Features', fontsize = 16)
        for feature, i in zip(df_old_cols, counts):
            first = sns.histplot(ax = axes[i, 0], x = df_old[feature], kde = True, bins = 30, color = 'deeppink')
            first.set_ylabel(f"Before transformation", fontsize = 15)
            second = sns.histplot(ax = axes[i, 1], x = df_new[feature], kde = True, bins = 30, color = 'dodgerblue')
            second.set_ylabel(f"After transformation", fontsize = 15)
        plt.savefig('./plots/histograms.png', facecolor = 'white', transparent = False)
        #plt.show()
    """

    def histogram_plots(self, df_old, df_new):

        """ Histograms of features before and after transformation """

        cols = 12
        rows = round(df_old.shape[1]/cols)
        sns.set(rc={"figure.figsize": (45, 20)})
        fig1, ax1 = plt.subplots(rows, cols)
        fig1.subplots_adjust(top = 0.96)
        fig1.suptitle('Histograms Before Transformations', fontsize = 22)
        for ax1, feature in zip(ax1.flat, df_old.columns):
            h = sns.histplot(df_old[feature], kde = True, bins = 30, color = 'gold', ax = ax1)
            h.set_ylabel('')
            h.set_xlabel(feature, fontsize = 13)
        fig1.savefig('./plots/histograms-before.png', facecolor = 'white', transparent = False)
        fig2, ax2 = plt.subplots(rows, cols)
        fig2.subplots_adjust(top = 0.96)
        fig2.suptitle('Histograms After Transformations', fontsize = 22)
        df_new = df_new[df_old.columns]
        for ax2, feature in zip(ax2.flat, df_new.columns):
            h = sns.histplot(df_new[feature], kde = True, bins = 30, color = 'deeppink', ax = ax2)
            h.set_ylabel('')
            h.set_xlabel(feature, fontsize = 13)
        fig2.savefig('./plots/histograms-after.png', facecolor = 'white', transparent = False)
        #plt.show()
    
    def interaction_plots(self, df):

        """ Interaction plot of chosen features """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Interaction Plot', fontsize = 14)
        fig = interaction_plot(
            x = df['MoSold'],
            trace = df['SaleType'],
            response = df['SalePrice'],
            ms = 10,
            ax = ax)
        fig.savefig('./plots/interaction.png', facecolor = 'white', transparent = False)
    
    def transformation(self, df):
        
        """ Finds the best feature transformation functions using box-cox """

        # calculate skewness for each feature
        skew_vals_base = df.agg(['skew', 'kurtosis']).transpose()
        skewness = skew_vals_base[abs(skew_vals_base['skew']) > 0.5].sort_values('skew', ascending = False)
        skewness_cols = skewness.index.tolist()
        # proceed with only those columns that have skewness > 0.5
        to_transform = df[skewness_cols].reset_index(drop = True)
        to_transform_cols = to_transform.columns.tolist()
        # keep features where skewness is < 0.5
        df1 = df.drop(to_transform_cols, axis = 1).reset_index(drop = True)
        bcx_target_list = []
        bcx_lambda_list = []
        for i, column in enumerate(to_transform_cols, 1):
            bcx_target, bcx_lambda = boxcox(to_transform[column] + 1)
            bcx_target_list.append(bcx_target)
            bcx_lambda_list.append(bcx_lambda)
        # combine tranformed features with non-transformed
        transformed_df = pd.DataFrame(bcx_target_list).T
        transformed_df.columns = to_transform.columns
        train_features = pd.concat([transformed_df, df1], axis = 1)
        # columns that are NOT good for any transformations
        cols = ['CentralAir', 'LotShape', 'YearRemodAdd', 'PavedDrive', 'GarageCond', 'YearBuilt']
        temp = df[cols].reset_index(drop = True)
        train_features1 = train_features.drop(cols, axis = 1)
        train_features_final = pd.concat([train_features1, temp], axis = 1)
        print('<<< Feature transformation applied (', train_features.shape[1],  'features ) >>>')
        return to_transform, train_features_final
    
    def polynomial_features(self, x_df, y, degree):
        
        """ Generates polynomial features up to N degree """
        df_list = []
        for i in range(degree):
            n_df = np.power(x_df, degree)
            n_df = n_df.add_suffix('^'+str(degree))
            df_list.append(n_df)
            degree = degree - 1
        strn = ''
        for i in range(len(df_list)):
            strn += 'df_list[' + str(i) + ']' + ','
            #strn = strn.
            #poly_df = pd.concat([df_list[i], y], axis = 1)
        strn.pop(-1)
        print(strn)
        poly_df = pd.concat([df_list[0], df_list[1], df_list[2], y], axis = 1)
        #squared_df = np.power(x_df, 2)
        #squared_df = squared_df.add_suffix('^2')
        #cubed_df = np.power(x_df, 3)
        #cubed_df = cubed_df.add_suffix('^3')
        #new_df = pd.concat([x_df, squared_df, cubed_df, y], axis = 1)
        return poly_df
    
    def vif_calc(self, df):
        
        """ Calculates a Variance Inflation Factor for each feature in the given dataframe """

        X = df.loc[:, df.columns != 'SalePrice']
        y = df['SalePrice']
        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        vif_data.sort_values(by = ['VIF'])
        return vif_data
        
    def pca_calc(self, df_x, n_comp):
        
        """ Applies feature scaling and performs PCA """

        rc = RobustScaler()
        X_scaled = rc.fit_transform(df_x)
        # Initial PCA to choose the number of components
        pca = PCA(n_components = 30)
        pca.fit(X_scaled)
        # Visualization
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
        ax.set_title('PCA Components and Explained Variance', fontsize = 15)
        ax.set_xlabel('Number of components (Dimensions)', fontsize = 11)
        ax.set_ylabel('Explained variance (%)', fontsize = 11)
        fig.savefig('./plots/pca-components.png', facecolor = 'white', transparent = False)
        #plt.show()
        # Apply PCA with predefined n to the whole dataset
        pca = PCA(n_components = n_comp, svd_solver = 'full')
        X_pca = pca.fit_transform(X_scaled)
        X_pca = pd.DataFrame(X_pca)
        #print(np.cumsum(pca.explained_variance_ratio_ * 100))
        print('<<< PCA with', X_pca.shape[1], 'components applied >>>')
        return X_pca
        
    def train_data_preprocessing(self):
        
        """ Performs the full preprocessing of the training dataset """
        
        print('\n======== Initial Data Loading and Preprocessing ========\n')

        """ .csv datasets load """

        train_df = pd.read_csv(self.train_csv_dir)
        #print('Training dataset has', train_df.shape[0], 'rows and', train_df.shape[1], 'columns.\n')
        
        """ Drop ID """
        
        train_df = train_df.drop(['Id'], axis = 1)
        
        """ Separate data with numerical values """

        train_df_num = train_df.select_dtypes(include=[np.number])
        train_df = train_df.drop([col for col in train_df_num], axis = 1)
        
        """ Missing Values (numerical features only) """
        
        perc_missing_num = train_df_num.isnull().sum() * 100 / len(train_df_num)
        nans_num = pd.DataFrame({'col_name': train_df_num.columns, 'percent_missing': perc_missing_num})
        
        """ Impute Missing Values """
        
        train_df_num = train_df_num.drop(columns=['GarageYrBlt']) # drop GarageYrBlt
        train_df_num['MasVnrArea'] = train_df_num['MasVnrArea'].fillna(0)
        train_df_num = train_df_num.drop(columns=['LotFrontage']) # drop LotFrontage
        
        """ Numerical Feature Simplification """
        
        for col in [['OverallQual', 'OverallCond']]:
            train_df_num[col] = train_df_num[col].astype('object')
            
        train_df_num['MSSubClass'].replace(20, '1 story 1946+', inplace = True)
        train_df_num['MSSubClass'].replace(30, '1 story 1945-', inplace = True)
        train_df_num['MSSubClass'].replace(40, '1 story unf attic', inplace = True)
        train_df_num['MSSubClass'].replace(45, '1,5 story unf', inplace = True)
        train_df_num['MSSubClass'].replace(50, '1,5 story fin', inplace = True)
        train_df_num['MSSubClass'].replace(60, '2 story 1946+', inplace = True)
        train_df_num['MSSubClass'].replace(70, '2 story 1945-', inplace = True)
        train_df_num['MSSubClass'].replace(75, '2,5 story all ages', inplace = True)
        train_df_num['MSSubClass'].replace(80, 'split/multi level', inplace = True)
        train_df_num['MSSubClass'].replace(85, 'split foyer', inplace = True)
        train_df_num['MSSubClass'].replace(90, 'duplex all style/age', inplace = True)
        train_df_num['MSSubClass'].replace(120, '1 story PUD 1946+', inplace = True)
        train_df_num['MSSubClass'].replace(150, '1,5 story PUD all', inplace = True)
        train_df_num['MSSubClass'].replace(160, '2 story PUD 1946+', inplace = True)
        train_df_num['MSSubClass'].replace(180, 'PUD multilevel', inplace = True)
        train_df_num['MSSubClass'].replace(190, '2 family conversion', inplace = True)
        train_df = pd.concat([train_df, train_df_num], axis = 1)
        
        """ Categorical Features """
        
        train_df_cat = train_df.select_dtypes(include=[object])
        train_df = train_df.drop([col for col in train_df_cat], axis = 1)
        
        """ Missing Values (categorical features only) """
        
        perc_missing_cat = train_df_cat.isnull().sum() * 100 / len(train_df_cat)
        nans_cat = pd.DataFrame({'col_name': train_df_cat.columns, 'percent_missing': perc_missing_cat})

        """ Barplots of all missing values """

        self.missing_vals_plots(nans_num, nans_cat)
        
        """ Impute Missing Values """
        
        train_df_cat['MasVnrType'] = train_df_cat['MasVnrType'].fillna('None')
        train_df_cat[['Fence', 'PoolQC', 'FireplaceQu', 'Alley', 'BsmtQual', \
        'BsmtCond', 'BsmtFinType1', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MiscFeature']] = \
        train_df_cat[['Fence', 'PoolQC', 'FireplaceQu', 'Alley', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', \
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MiscFeature']].fillna('NA')
        
        rand_val = np.random.choice(['No', 'Av', 'Gd', 'Mn'], 1)
        train_df_cat['BsmtExposure'] = train_df_cat['BsmtExposure'].fillna(rand_val[0])
        
        rand_val = np.random.choice(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'], 1)
        train_df_cat['BsmtFinType2'] = train_df_cat['BsmtFinType2'].fillna(rand_val[0])
        
        rand_val = np.random.choice(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], 1)
        train_df_cat['Electrical'] = train_df_cat['BsmtFinType2'].fillna(rand_val[0])
        
        """ Categorical Feature Simplification """
        
        train_df_cat['GarageQual'] = train_df_cat.GarageQual.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                     'Gd' : 3, 'Ex' : 3, 'TA' : 4 })
        train_df_cat['GarageCond'] = train_df_cat.GarageCond.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                             'Gd' : 3, 'Ex' : 3, 'TA' : 4 })
        train_df_cat['FireplaceQu'] = train_df_cat.FireplaceQu.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                             'Gd' : 3, 'Ex' : 3, 'TA' : 4 })
        train_df_cat['Functional'] = train_df_cat.Functional.replace({'Typ' : 4, 'Min1' : 3, 'Min2' : 3, \
                                                             'Mod': 2, 'Maj1' : 1, 'Maj2' : 1, 'Sev' : 1 })
        train_df_cat['HeatingQC'] = train_df_cat.HeatingQC.replace({'Po' : 1, 'Fa' : 1, 'Gd' : 2, 'Ex' : 2, 'TA' : 3 })
        train_df_cat['BsmtFinType1'] = train_df_cat.BsmtFinType1.replace({'NA' : 1, 'Unf' : 2, 'LwQ' : 2, \
                                                                 'Rec' : 3, 'BLQ' : 3, 'ALQ' : 3, 'GLQ' : 4 })
        train_df_cat['BsmtFinType2'] = train_df_cat.BsmtFinType2.replace({'NA' : 1, 'Unf' : 2, 'LwQ' : 2, \
                                                                 'Rec' : 3, 'BLQ' : 3, 'ALQ' : 3, 'GLQ' : 4 })
        train_df_cat['BsmtCond'] = train_df_cat.BsmtCond.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                         'TA' : 3, 'Gd' : 4 })
        train_df_cat['BsmtQual'] = train_df_cat.BsmtQual.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                         'TA' : 3, 'Gd' : 4, 'Ex': 4 })
        train_df_cat['ExterCond'] = train_df_cat.ExterCond.replace({'Fa' : 1, 'Po' : 1, \
                                                           'Gd' : 2, 'Ex' : 2, 'TA' : 3 })
        train_df_cat['ExterQual'] = train_df_cat.ExterQual.replace({'Fa' : 1, 'Po' : 1, \
                                                           'Gd' : 2, 'Ex' : 2, 'TA' : 3 })
        train_df_cat['LotConfig'] =  train_df_cat.LotConfig.replace({'FR3' : 1, 'FR2' : 1, \
                                                           'CulDSac' : 2, 'Corner' : 3, 'Inside' : 4 })
        train_df_cat['Condition1'] = train_df_cat.Condition1.replace({'RRNe' : 1, 'RRNn' : 1, \
                                                           'RRAe' : 1, 'RRAn' : 1, 'PosA' : 2, \
                                                            'PosN' : 2, 'Artery': 3,  'Feedr': 4,  'Norm': 5 })
        train_df_cat['Exterior1st'] = train_df_cat.Exterior1st.replace({'CBlock' : 1, 'ImStucc' : 1, \
                                                           'AsphShn' : 1, 'Stone' : 1, 'BrkComm' : 1,  'AsbShng' : 1, \
                                                            'Stucco': 1,  'WdShing': 1, 'BrkFace': 2 , 'CemntBd': 3, \
                                                            'Plywood': 4, 'Wd Sdng': 5, 'MetalSd': 6, 'HdBoard': 7, \
                                                            'VinylSd': 8})
        train_df_cat['Exterior2nd'] = train_df_cat.Exterior2nd.replace({'CBlock' : 1, 'ImStucc' : 1, 'AsphShn' : 1, \
                                                            'Stone' : 1, 'Brk Cmn' : 1,  'AsbShng' : 1, 'Other': 1, \
                                                            'Stucco': 1,  'Wd Shng': 1, 'BrkFace': 2 , 'CmentBd': 3, \
                                                            'Plywood': 4, 'Wd Sdng': 5, 'MetalSd': 6, 'HdBoard': 7, \
                                                            'VinylSd': 8})
        train_df_cat['SaleType'] = train_df_cat.SaleType.replace({'Con' : 1, 'Oth' : 1, 'CWD' : 1, 'ConLw' : 1, \
                                                           'ConLI' : 1,  'ConLD' : 1, 'COD' : 2,  'New' : 3, 'WD': 4})
        train_df_cat['Foundation'] = train_df_cat.Foundation.replace({'Wood' : 1, 'Stone' : 1, 'Slab' : 1, 'BrkTil' : 2, \
                                                           'CBlock' : 3,  'PConc' : 4})
        train_df_cat['OverallCond'] = train_df_cat.OverallCond.replace({1 : 1, 2 : 1, 3 : 2, 4 : 3, 5 : 4, 6 : 5, \
                                                           7 : 6, 8 : 7, 9 : 8, 10 : 9})
        train_df_cat['OverallQual'] = train_df_cat.OverallQual.replace({1 : 1, 2 : 1, 3 : 2, 4 : 3, 5 : 4, 6 : 5, \
                                                           7 : 6, 8 : 7, 9 : 8, 10 : 9})
        # convert categorical features above to category types
        for col in [['OverallQual', 'OverallCond', 'Foundation', 'SaleType', 'Exterior2nd', 'Exterior1st', \
        'Condition1', 'LotConfig', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType2', \
        'BsmtFinType1', 'HeatingQC', 'Functional', 'FireplaceQu', 'GarageCond', 'GarageQual']]:
            train_df_cat[col] = train_df_cat[col].astype('object')
        
        """ Drop Some Constant Features"""

        train_df_cat.drop('Street', axis = 1, inplace = True)
        train_df_cat.drop('Utilities', axis = 1, inplace = True)
        train_df_cat.drop('PoolQC', axis = 1, inplace = True)
        train_df_cat.drop('MiscFeature', axis = 1, inplace = True)
        train_df_cat.drop('Condition2', axis = 1, inplace = True)
        train_df_cat.drop('RoofMatl', axis = 1, inplace = True)
        train_df_cat.drop('Heating', axis = 1, inplace = True)
        
        train_df = pd.concat([train_df, train_df_cat], axis = 1)
        
        """ Simplification of Other Features That Involves Deletion of Rows"""
        
        train_df.drop(train_df[train_df['Neighborhood'] == 'Blueste'].index, inplace = True)
        train_df.drop(train_df[train_df['RoofStyle'] == 'Shed'].index, inplace = True)
        train_df.drop(train_df[train_df['SaleCondition'] == 'AdjLand'].index, inplace = True)
        train_df.drop(train_df[train_df['MSSubClass'] == '1 story unf attic'].index, inplace = True)
        train_df = train_df.reset_index(drop = True)
        print('<<< Imputation and feature simplification applied >>>')
        
        """ Label encoding/factorizing the remaining categorical variables """
        
        cat_cols = train_df.select_dtypes(include=['object'])
        train_df = DfLabelEncoder(columns = cat_cols.columns).fit_transform(train_df)
        print('<<< Feature encoding applied >>>')
        #print('There are', len(train_df), 'observations,',train_df.select_dtypes(include=['object']).shape[1], 'categorical features and', \
        #train_df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).shape[1], 'numerical features.')
        
        """ Feature Engineering """
        
        """ # Uncomment to create new features """
        train_df['Total_Bathrooms'] = train_df['BsmtFullBath'] + train_df['FullBath'] + (0.5 * train_df['BsmtHalfBath']) + (0.5 * train_df['HalfBath'])
        train_df['Ground_SF'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
        train_df['Floor_SF'] = train_df['1stFlrSF'] + train_df['2ndFlrSF']
        train_df['Total_Porch_SF'] = train_df['OpenPorchSF'] + train_df['EnclosedPorch'] + train_df['3SsnPorch'] + train_df['ScreenPorch']
        train_df['Age'] = train_df['YrSold'] - train_df['YearRemodAdd']
        train_df['YearBuiltPlusYearRemodAdd'] = train_df['YearBuilt'] + train_df['YearRemodAdd']
        train_df['Total_Sq_Footage'] = train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']

        """ Interaction Terms """
        
        """
        # Uncomment to create interaction terms 
        train_df['GarageArea_x_GarageCars'] = train_df['GarageArea'] * train_df['GarageCars']
        train_df['GarageArea_x_GarageQual'] = train_df['GarageArea'] * train_df['GarageQual']
        train_df['GarageArea_x_GarageCond'] = train_df['GarageArea'] * train_df['GarageCond']
        train_df['GarageQual_x_GarageCond'] = train_df['GarageQual'] * train_df['GarageCond']
        train_df['OverallQual_x_OverallCond'] = train_df['OverallQual'] * train_df['OverallCond']
        train_df['ExterQual_x_ExterCond'] = train_df['ExterQual'] * train_df['ExterCond']
        train_df['GrLivArea_x_TotRmsAbvGrd'] = train_df['GrLivArea'] * train_df['TotRmsAbvGrd']
        train_df['BsmtFinSF1_x_BsmtFinType1'] = train_df['BsmtFinSF1'] * train_df['BsmtFinType1']
        train_df['MasVnrArea_x_MasVnrType'] = train_df['MasVnrArea'] * train_df['MasVnrType']
        """
        
        """ Polynomial Features """
        
        """ 
        # Uncomment to create a higher order terms 
        train_df['ExterQual^2'] = train_df['ExterQual'] ** 2
        train_df['GarageType^2'] = train_df['GarageType'] ** 2
        train_df['2ndFlrSF^2'] = train_df['2ndFlrSF'] ** 2
        train_df['BsmtFinType1^2'] = train_df['BsmtFinType1'] ** 2
        train_df['BsmtQual^2'] = train_df['BsmtQual'] ** 2
        train_df['Foundation^2'] = train_df['Foundation'] ** 2
        
        train_df['(ExterQual_x_ExterCond)^2'] = train_df['ExterQual_x_ExterCond'] ** 2
        train_df['ExterQual^2_x_ExterCond'] = train_df['ExterQual^2'] * train_df['ExterCond']
        train_df['(ExterQual^2_x_ExterCond)^2'] = train_df['ExterQual^2_x_ExterCond'] ** 2
        train_df['ExterQual_x_ExterQual^2_x_ExterCond'] = train_df['ExterQual'] * train_df['ExterQual^2'] * train_df['ExterCond']
        train_df['(BsmtFinSF1_x_BsmtFinType1)^2'] = train_df['BsmtFinSF1_x_BsmtFinType1'] ** 2
        train_df['BsmtFinSF1_x_BsmtFinType1^2'] = train_df['BsmtFinSF1'] * train_df['BsmtFinType1^2']
        train_df['BsmtFinSF1_x_BsmtFinType1_x_BsmtFinType1^2'] = train_df['BsmtFinSF1'] * train_df['BsmtFinType1'] * train_df['BsmtFinType1^2']
        train_df['(BsmtFinSF1_x_BsmtFinType1_x_BsmtFinType1^2)^2'] = train_df['BsmtFinSF1_x_BsmtFinType1_x_BsmtFinType1^2'] ** 2 
        """
        print('<<< Feature engineering is applied >>>')
        
        # Compute correlations
        train_features = self.correlations(train_df, train_df['SalePrice'], 0)
        
        # Create scatter plots
        self.scatter_plots(train_features, order = 1)

        # Create box plots
        self.box_plots(train_features)

        # Create a single interaction plot
        self.interaction_plots(train_features)
        
        # Remove outliers
        train_features = self.outlier_removal(train_features)
        
        # Transform features
        train_features_old, train_features_new = self.transformation(train_features)
        
        # Create histograms
        self.histogram_plots(train_features_old, train_features_new)
        
        print('\nTraining dataset has', train_features.shape[0], 'rows and', train_features.shape[1], 'columns.\n')
        
        # Compute VIF values
        
        vif = self.vif_calc(train_features)
        #print(vif)
        
        # Compute PCA
    
        # Uncomment to perform PCA and use components as inputs to a model 
        X = self.pca_calc(train_features.loc[:, train_features.columns != 'SalePrice'], n_comp = 25)
       
        """
        # Uncomment to use best features for OLS 
        X = train_features[['OverallQual', 'TotalBsmtSF', 'OverallQual_x_OverallCond', 'Age', 'GarageArea_x_GarageCars', 'Total_Bathrooms', 'Ground_SF', \
        'LotArea', 'Fireplaces', '1stFlrSF', 'YearBuilt', 'BsmtFinSF1_x_BsmtFinType1^2', 'BsmtFinType1^2', 'Foundation^2', 'WoodDeckSF']]
        """
        
        # Uncomment to use all features
        X = train_features.loc[:, train_features.columns != 'SalePrice']
        y = train_features['SalePrice'] 
        
        print('<<<', X.shape[1], 'features selected to perform modeling >>>\n')
        train_features_final = pd.concat([X, y], axis = 1)
        return train_features_final
    
    def test_data_preprocessing(self):
        
        """ Performs the full preprocessing of the testing dataset """
            
        print('\n======== Initial Data Loading and Preprocessing ========\n')
        
        """ .csv test dataset load """
        
        test_df = pd.read_csv(self.test_csv_dir)
        #print('Testing dataset has', test_df.shape[0], 'rows and', test_df.shape[1], 'columns.\n')
        
        """ Drop ID"""
        
        #test_id_col = test_df['Id'].tolist()
        test_df = test_df.drop(['Id'], axis = 1)
        
        """ Separate data with numerical values """

        test_df_num = test_df.select_dtypes(include = [np.number])
        test_df = test_df.drop([col for col in test_df_num], axis = 1)
        
        """ Impute Missing Values """
        
        test_df_num = test_df_num.drop(columns = ['GarageYrBlt']) # drop GarageYrBlt
        test_df_num['MasVnrArea'] = test_df_num['MasVnrArea'].fillna(0)
        test_df_num = test_df_num.drop(columns = ['LotFrontage']) # drop LotFrontage

        #KNN imputed
        imputer = KNNImputer(n_neighbors = 5, weights = 'distance', metric = 'nan_euclidean')
        test_df_num[['GarageArea', 'GarageCars', 'BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', \
        'BsmtFinSF1']] = imputer.fit_transform(test_df_num[['GarageArea', 'GarageCars', 'BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', \
        'BsmtFinSF1']])
        
        for col in [['OverallQual', 'OverallCond']]:
            test_df_num[col] = test_df_num[col].astype('object')
            
        test_df_num['MSSubClass'].replace(20, '1 story 1946+', inplace = True)
        test_df_num['MSSubClass'].replace(30, '1 story 1945-', inplace = True)
        test_df_num['MSSubClass'].replace(40, '1 story unf attic', inplace = True)
        test_df_num['MSSubClass'].replace(45, '1,5 story unf', inplace = True)
        test_df_num['MSSubClass'].replace(50, '1,5 story fin', inplace = True)
        test_df_num['MSSubClass'].replace(60, '2 story 1946+', inplace = True)
        test_df_num['MSSubClass'].replace(70, '2 story 1945-', inplace = True)
        test_df_num['MSSubClass'].replace(75, '2,5 story all ages', inplace = True)
        test_df_num['MSSubClass'].replace(80, 'split/multi level', inplace = True)
        test_df_num['MSSubClass'].replace(85, 'split foyer', inplace = True)
        test_df_num['MSSubClass'].replace(90, 'duplex all style/age', inplace = True)
        test_df_num['MSSubClass'].replace(120, '1 story PUD 1946+', inplace = True)
        test_df_num['MSSubClass'].replace(150, '1,5 story PUD all', inplace = True)
        test_df_num['MSSubClass'].replace(160, '2 story PUD 1946+', inplace = True)
        test_df_num['MSSubClass'].replace(180, 'PUD multilevel', inplace = True)
        test_df_num['MSSubClass'].replace(190, '2 family conversion', inplace = True)
        test_df = pd.concat([test_df, test_df_num], axis = 1)
        
        """ Categorical Features """
            
        test_df_cat = test_df.select_dtypes(include=[object])
        test_df = test_df.drop([col for col in test_df_cat], axis = 1)
        
        
        """ Missing Values (categorical features only) """
        
        percent_missing_nans = test_df_cat.isnull().sum() * 100 / len(test_df_cat)
        nans = pd.DataFrame({'col_name': test_df_cat.columns, 'percent_missing': percent_missing_nans})
        
        test_df_cat['MasVnrType'] = test_df_cat['MasVnrType'].fillna('None')
        test_df_cat[['Fence', 'PoolQC', 'FireplaceQu', 'Alley', 'BsmtQual', \
        'BsmtCond', 'BsmtFinType1', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MiscFeature']] = \
        test_df_cat[['Fence', 'PoolQC', 'FireplaceQu', 'Alley', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', \
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MiscFeature']].fillna('NA')
        
        rand_val = np.random.choice(['Gd', 'Av', 'Mn', 'No', 'NA'], 1)
        test_df_cat['BsmtExposure'] = test_df_cat['BsmtExposure'].fillna(rand_val[0])
        
        rand_val = np.random.choice(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], 1)
        test_df_cat['BsmtFinType2'] = test_df_cat['BsmtFinType2'].fillna(rand_val[0])
        
        rand_val = np.random.choice(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], 1)
        test_df_cat['Electrical'] = test_df_cat['BsmtFinType2'].fillna(rand_val[0])
        
        options = ['C (all)', 'FV', 'RH', 'RL', 'RM']
        rand_val = np.random.choice(options, 1)
        test_df_cat['MSZoning'] = test_df_cat['MSZoning'].fillna(rand_val[0])
        
        options = ['VinylSd', 'MetalSd', 'HdBoard', 'Wd Sdng', 'Plywood', 'CemntBd', 'BrkFace', 'WdShing', \
        'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'CBlock']
        rand_val = np.random.choice(options, 1)
        test_df_cat['Exterior1st'] = test_df_cat['Exterior1st'].fillna(rand_val[0])
        test_df_cat['Exterior2nd'] = test_df_cat.Exterior2nd.replace({'CmentBd' : 'CemntBd', 'Wd Shng' : 'WdShing',
                                                                        'Brk Cmn' : 'BrkComm'})
        test_df_cat['Exterior2nd'] = test_df_cat['Exterior2nd'].fillna(rand_val[0])
        
        options = ['Ex', 'Gd', 'TA', 'Fa']
        rand_val = np.random.choice(options, 1)
        test_df_cat['KitchenQual'] = test_df_cat['KitchenQual'].fillna(rand_val[0])
        
        options = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev']
        rand_val = np.random.choice(options, 1)
        test_df_cat['Functional'] = test_df_cat['Functional'].fillna(rand_val[0])
        
        options = ['WD', 'New', 'COD', 'CWD', 'ConLD', 'ConLI', 'Con', 'ConLw', 'Oth']
        rand_val = np.random.choice(options, 1)
        test_df_cat['SaleType'] = test_df_cat['SaleType'].fillna(rand_val[0])
        
        """ Categorical Feature Simplification """
        
        test_df_cat['GarageQual'] = test_df_cat.GarageQual.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                     'Gd' : 3, 'Ex' : 3, 'TA' : 4 })
        test_df_cat['GarageCond'] = test_df_cat.GarageCond.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                             'Gd' : 3, 'Ex' : 3, 'TA' : 4 })
        test_df_cat['FireplaceQu'] = test_df_cat.FireplaceQu.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                             'Gd' : 3, 'Ex' : 3, 'TA' : 4 })
        test_df_cat['Functional'] = test_df_cat.Functional.replace({'Typ' : 4, 'Min1' : 3, 'Min2' : 3, \
                                                             'Mod': 2, 'Maj1' : 1, 'Maj2' : 1, 'Sev' : 1 })
        test_df_cat['HeatingQC'] = test_df_cat.HeatingQC.replace({'Po' : 1, 'Fa' : 1, 'Gd' : 2, 'Ex' : 2, 'TA' : 3 })
        test_df_cat['BsmtFinType1'] = test_df_cat.BsmtFinType1.replace({'NA' : 1, 'Unf' : 2, 'LwQ' : 2, \
                                                                 'Rec' : 3, 'BLQ' : 3, 'ALQ' : 3, 'GLQ' : 4 })
        test_df_cat['BsmtFinType2'] = test_df_cat.BsmtFinType2.replace({'NA' : 1, 'Unf' : 2, 'LwQ' : 2, \
                                                                 'Rec' : 3, 'BLQ' : 3, 'ALQ' : 3, 'GLQ' : 4 })
        test_df_cat['BsmtCond'] = test_df_cat.BsmtCond.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                         'TA' : 3, 'Gd' : 4 })
        test_df_cat['BsmtQual'] = test_df_cat.BsmtQual.replace({'NA' : 1, 'Fa' : 2, 'Po' : 2, \
                                                         'TA' : 3, 'Gd' : 4, 'Ex': 4 })
        test_df_cat['ExterCond'] = test_df_cat.ExterCond.replace({'Fa' : 1, 'Po' : 1, \
                                                           'Gd' : 2, 'Ex' : 2, 'TA' : 3 })
        test_df_cat['ExterQual'] = test_df_cat.ExterQual.replace({'Fa' : 1, 'Po' : 1, \
                                                           'Gd' : 2, 'Ex' : 2, 'TA' : 3 })
        test_df_cat['LotConfig'] =  test_df_cat.LotConfig.replace({'FR3' : 1, 'FR2' : 1, \
                                                           'CulDSac' : 2, 'Corner' : 3, 'Inside' : 4 })
        test_df_cat['Condition1'] = test_df_cat.Condition1.replace({'RRNe' : 1, 'RRNn' : 1, \
                                                           'RRAe' : 1, 'RRAn' : 1, 'PosA' : 2, \
                                                            'PosN' : 2, 'Artery': 3,  'Feedr': 4,  'Norm': 5 })
        test_df_cat['Exterior1st'] = test_df_cat.Exterior1st.replace({'CBlock' : 1, 'ImStucc' : 1, \
                                                           'AsphShn' : 1, 'Stone' : 1, 'BrkComm' : 1,  'AsbShng' : 1, \
                                                            'Stucco': 1,  'WdShing': 1, 'BrkFace': 2 , 'CemntBd': 3, \
                                                            'Plywood': 4, 'Wd Sdng': 5, 'MetalSd': 6, 'HdBoard': 7, \
                                                            'VinylSd': 8})
        test_df_cat['Exterior2nd'] = test_df_cat.Exterior2nd.replace({'CBlock' : 1, 'ImStucc' : 1, 'AsphShn' : 1, \
                                                            'Stone' : 1, 'BrkComm' : 1,  'AsbShng' : 1, 'Other': 1, \
                                                            'Stucco': 1,  'WdShing': 1, 'BrkFace': 2 , 'CemntBd': 3, \
                                                            'Plywood': 4, 'Wd Sdng': 5, 'MetalSd': 6, 'HdBoard': 7, \
                                                            'VinylSd': 8})
        test_df_cat['SaleType'] = test_df_cat.SaleType.replace({'Con' : 1, 'Oth' : 1, 'CWD' : 1, 'ConLw' : 1, \
                                                           'ConLI' : 1,  'ConLD' : 1, 'COD' : 2,  'New' : 3, 'WD': 4})
        test_df_cat['Foundation'] = test_df_cat.Foundation.replace({'Wood' : 1, 'Stone' : 1, 'Slab' : 1, 'BrkTil' : 2, \
                                                           'CBlock' : 3,  'PConc' : 4})
        test_df_cat['OverallCond'] = test_df_cat.OverallCond.replace({1 : 1, 2 : 1, 3 : 2, 4 : 3, 5 : 4, 6 : 5, \
                                                           7 : 6, 8 : 7, 9 : 8, 10 : 9})
        test_df_cat['OverallQual'] = test_df_cat.OverallQual.replace({1 : 1, 2 : 1, 3 : 2, 4 : 3, 5 : 4, 6 : 5, \
                                                           7 : 6, 8 : 7, 9 : 8, 10 : 9})
        # convert categorical features above to category types
        for col in [['OverallQual', 'OverallCond', 'Foundation', 'SaleType', 'Exterior2nd', 'Exterior1st', \
        'Condition1', 'LotConfig', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType2', 'BsmtFinType1', \
        'HeatingQC', 'Functional', 'FireplaceQu', 'GarageCond', 'GarageQual']]:
            test_df_cat[col] = test_df_cat[col].astype('object')
            
        """ Drop Some Constant Features"""

        test_df_cat.drop('Street', axis = 1, inplace = True)
        test_df_cat.drop('Utilities', axis = 1, inplace = True)
        test_df_cat.drop('PoolQC', axis = 1, inplace = True)
        test_df_cat.drop('MiscFeature', axis = 1, inplace = True)
        test_df_cat.drop('Condition2', axis = 1, inplace = True)
        test_df_cat.drop('RoofMatl', axis = 1, inplace = True)
        test_df_cat.drop('Heating', axis = 1, inplace = True)
        print('<<< Imputation and feature simplification applied >>>')
        test_df = pd.concat([test_df, test_df_cat], axis = 1)
        
        """ Label encoding/factorizing the remaining categorical variables """
        
        cat_cols = test_df.select_dtypes(include=['object'])
        test_df = DfLabelEncoder(columns = cat_cols.columns).fit_transform(test_df)
        print('<<< Feature encoding applied >>>')
        
        """ Feature Engineering """
        
        """ # Uncomment to create new features """
        test_df['Total_Bathrooms'] = test_df['BsmtFullBath'] + test_df['FullBath'] + (0.5 * test_df['BsmtHalfBath']) + (0.5 * test_df['HalfBath'])
        test_df['Ground_SF'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']
        test_df['Floor_SF'] = test_df['1stFlrSF'] + test_df['2ndFlrSF']
        test_df['Total_Porch_SF'] = test_df['OpenPorchSF'] + test_df['EnclosedPorch'] + test_df['3SsnPorch'] + test_df['ScreenPorch']
        test_df['Age'] = test_df['YrSold'] - test_df['YearRemodAdd']
        test_df['YearBuiltPlusYearRemodAdd'] = test_df['YearBuilt'] + test_df['YearRemodAdd']
        test_df['Total_Sq_Footage'] = test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

        
        """ Interaction Terms """
        
        """ Uncomment to create interaction terms 
        test_df['GarageArea_x_GarageCars'] = test_df['GarageArea'] * test_df['GarageCars']
        test_df['GarageArea_x_GarageQual'] = test_df['GarageArea'] * test_df['GarageQual']
        test_df['GarageArea_x_GarageCond'] = test_df['GarageArea'] * test_df['GarageCond']
        test_df['GarageQual_x_GarageCond'] = test_df['GarageQual'] * test_df['GarageCond']
        test_df['OverallQual_x_OverallCond'] = test_df['OverallQual'] * test_df['OverallCond']
        test_df['ExterQual_x_ExterCond'] = test_df['ExterQual'] * test_df['ExterCond']
        test_df['GrLivArea_x_TotRmsAbvGrd'] = test_df['GrLivArea'] * test_df['TotRmsAbvGrd']
        test_df['BsmtFinSF1_x_BsmtFinType1'] = test_df['BsmtFinSF1'] * test_df['BsmtFinType1']
        test_df['MasVnrArea_x_MasVnrType'] = test_df['MasVnrArea'] * test_df['MasVnrType']
        test_df['YearBuiltPlusYearRemodAdd'] = test_df['YearBuilt'] + test_df['YearRemodAdd']
        test_df['Total_Sq_Footage'] = test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
        """
        
        """ Polynomial Features """
        
        """ 
        # Uncomment to create a higher order terms
        
        test_df['ExterQual^2'] = test_df['ExterQual'] ** 2
        test_df['GarageType^2'] = test_df['GarageType'] ** 2
        test_df['2ndFlrSF^2'] = test_df['2ndFlrSF'] ** 2
        test_df['BsmtFinType1^2'] = test_df['BsmtFinType1'] ** 2
        test_df['BsmtQual^2'] = test_df['BsmtQual'] ** 2
        test_df['Foundation^2'] = test_df['Foundation'] ** 2
    
        test_df['(ExterQual_x_ExterCond)^2'] = test_df['ExterQual_x_ExterCond'] ** 2
        test_df['ExterQual^2_x_ExterCond'] = test_df['ExterQual^2'] * test_df['ExterCond']
        test_df['(ExterQual^2_x_ExterCond)^2'] = test_df['ExterQual^2_x_ExterCond'] ** 2
        test_df['ExterQual_x_ExterQual^2_x_ExterCond'] = test_df['ExterQual'] * test_df['ExterQual^2'] * test_df['ExterCond']
        test_df['(BsmtFinSF1_x_BsmtFinType1)^2'] = test_df['BsmtFinSF1_x_BsmtFinType1'] ** 2
        test_df['BsmtFinSF1_x_BsmtFinType1^2'] = test_df['BsmtFinSF1'] * test_df['BsmtFinType1^2']
        test_df['BsmtFinSF1_x_BsmtFinType1_x_BsmtFinType1^2'] = test_df['BsmtFinSF1'] * test_df['BsmtFinType1'] * test_df['BsmtFinType1^2']
        test_df['(BsmtFinSF1_x_BsmtFinType1_x_BsmtFinType1^2)^2'] = test_df['BsmtFinSF1_x_BsmtFinType1_x_BsmtFinType1^2'] ** 2
        """
        print('<<< Feature engineering applied >>>')
        
        """ PCA """
        
        """ # Uncomment to apply PCA and generate components 
        test_df = self.pca_calc(test_df)
        """
        print('\nTesting dataset has', test_df.shape[0], 'rows and', test_df.shape[1], 'columns.\n')
        return test_df