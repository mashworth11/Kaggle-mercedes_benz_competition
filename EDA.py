# -*- coding: utf-8 -*-
"""
Script for performing data analysis on the Merced-Benz data. 

Key things to look out for according to the winning solution 
(https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/37700):
    
    1) What are the data characteristics?
    2) How is our target data split?
    3) What is the process behind the data and what could affect the response?    
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')


#%% Data summaries (locations and variability)

# a first look at the data
pd.options.display.max_columns = 999
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
train_df.head()
#train_df.describe() # very useful for getting summary statistics (central tendancies, spread) on each variable
# =============================================================================
# Observations: We have a target (continuous var.) for time in seconds, anonymised
# features, which seem to be a mixture of categorical (possibly ordinal) and
# binary categorical. We also have an ID column, which isn't equal to the row number
# and could therefore be significant. 
# =============================================================================

# data types
dtype_df = train_df.dtypes.reset_index() # reset_index() labels each row with an index and converts to a column
dtype_df.columns = ["Count", "Column type"]
print(dtype_df.groupby("Column type").aggregate('count').reset_index())
# =============================================================================
# Observations: Indeed we can confirm majority are integers, 8 are 
# categorical and 1 is a float (continuous). May just want to double sure that
# the integer columns are binaries as we suspect. 
# =============================================================================

# missing values 
missing_df = train_df.isnull().sum(axis=0).reset_index() # row wise summation (i.e. sum along columns)
missing_df.columns = ["column_name", "missing_count"]
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by = "missing_count")
# =============================================================================
# Observations: Seems to be a good quality data set, with no missing values. Lit.
# =============================================================================



#%% Univariate target analysis

# TARGET
# scatter plot
plt.figure()
plt.scatter(range(train_df.shape[0]), np.sort(train_df['y'].values))
plt.xlabel('index')
plt.ylabel('time (s)')
# =============================================================================
# Observations: All but one data point (at 265) are below 180 seconds. 
# =============================================================================


# frequency plot of target
plt.figure()
train_df['y'].plot.hist(density = False, xlim = [0, 180], bins = 50)
plt.xlabel('time (s)')
# =============================================================================
# Observations: Distribution seems to be fairly normal here.
# =============================================================================

# alternatively could use seaborn
#plt.figure()
#sns.distplot(train_df['y'], bins = 50, kde = False)
#plt.xlabel('time (s)')


#%% Bivariate / feature analysis

# bivariate analysis on the categorical variables
def bivariate_plotter(var_name, plot_type):
    col_order = np.sort(train_df[var_name].unique()).tolist()
    #plt.figure()
    plot_type(x = var_name, y = 'y', data = train_df, order = col_order,  color = 'grey')
    plt.xlabel(var_name)
    plt.ylabel('time (s)')
    plt.title(f'Distribution of y (time in s) with {var_name}')
    
bivariate_plotter("X0", sns.stripplot)
bivariate_plotter("X3", sns.boxplot)
# =============================================================================
# Observations: In general, across the variables, average test time seems to be around 100s. 
# =============================================================================

# univariate analysis for the binary variables


# Stuff to do for next time:
# 1) how to establish whether ID is significant or not? Done
# 2) univariate analysis on categorical features see: https://www.kaggle.com/anokas/mercedes-eda-xgboost-starter-0-55
# 3) what is the importance of feature interactions? as suggested by winner
# 4) how to know if data is noisy and what is the effect?
# 5) how to conduct feature selection / dimensionality reduction?













