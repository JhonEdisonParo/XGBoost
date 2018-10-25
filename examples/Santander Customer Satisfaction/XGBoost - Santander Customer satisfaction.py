# Jhon Paredes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

sns.set(style="white", color_codes=True)

# Loading Train and Test Data 
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
sample = pd.read_csv('Data/sample_submission.csv')


#==============================================================================
#                Exploratory Data Analysis - EDA
#==============================================================================
def print_pretty_table(data):
    print(tabulate(data, headers='keys', tablefmt='psql'))
    
def brief_analysis(_column, xlabel="", ylabel="", title=""):
    m = len(_column)
    unique_values = list(_column.unique())
    
    if len(unique_values) <= 10:
        # Variable Categorica
        df = pd.DataFrame(_column.value_counts())
        df['Percentage'] = 100*df.iloc[:,0]/m
        print_pretty_table(df)
        _column.value_counts(dropna=False).plot(kind='barh', grid=True ); 
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.title(title)
        plt.show()
        plt.pause(0.01)

    elif len (unique_values) <= 100:
        df = pd.DataFrame(_column.value_counts())
        df['Percentage'] = 100*df.iloc[:,0]/m
        print("Printing Top-5 most common values from %i values" % len(_column.unique()) )
        print_pretty_table(df[:5])
        _column.value_counts(dropna=False)[:5].plot('bar'); 
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.title(title)
        plt.show()
        plt.pause(0.01)

    else:
        df = pd.DataFrame(_column.value_counts())
        df['Percentage'] = 100*df.iloc[:,0]/m
        print("Printing Top-5 most common values from %i values" % len(_column.unique()) )
        print_pretty_table(df[:5])
        _column.value_counts(dropna=False)[:5].plot('bar'); 
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.title(title)
        plt.show()
        plt.pause(0.01)
        print("COLUMN MORE THAN ONE VARIABLE..")
    
# TARGET == 0 : happy customers 
# TARGET == 1 : unhappy custormers
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
print("\n======================================================================")
print("\n== TARGET ==\nA little less then 4% are unhappy => unbalanced dataset\n")
print_pretty_table(df)
train.TARGET.value_counts(dropna=False).plot(kind='barh', grid=True ); 
plt.pause(0.01)


# VAR3
df = pd.DataFrame(train.var3.value_counts())
df['Percentage'] = 100*df['var3']/train.shape[0]
print("\n======================================================================")
print("\n== VAR3 == : nationality of the customer\n\nPrinting Top-5 most common values from %i values and" % len(train.var3.unique()) )
print("116 values in column var3 are -999999,it would mean that \nthe nationality of the customer is unknown %s\n" % str(train.loc[train.var3==-999999].shape))
# print_pretty_table(df[:5].transpose())
print_pretty_table(df[:5])
train.var3.value_counts(dropna=False)[:5].plot('bar'); 
plt.pause(0.01)
# Replace -999999 in VAR3 column with most common value 2 
train = train.replace(-999999, 2)
train.loc[train.var3==-999999].shape

brief_analysis(train.var3)


# NUM_VAR4
print("\n======================================================================")
print("\n== NUM_VAR4 == :\nnum_var4 is the number of products. Let's plot the distribution:")
xlabel = 'Number of bank products'
ylabel = 'Number of customers in train'
title = 'Most customers have 1 product with the bank'
brief_analysis(train.num_var4, xlabel, ylabel, title )


# Let's look at the density of the of happy/unhappy customers in function of the number of bank products
sns.FacetGrid(train, hue="TARGET", size=6).map(plt.hist, "num_var4").add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()

train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');
# brief_analysis(train[train.TARGET==1].num_var4, title='Amount of unhappy customers in function of the number of products')

# VAR38
print("\n======================================================================")
print("\n== NUM_VAR38 == :\nnum_var4 is the number of products. Let's plot the distribution:")
train.var38.describe()
train.loc[train['TARGET']==1, 'var38'].describe()
train.var38.hist(bins=1000);
plt.show()
train.var38.map(np.log).hist(bins=1000);
plt.show()
# where is the spike between 11 and 12  in the log plot ?
print("MODE: ", train.var38.map(np.log).mode())

# What are the most common values for var38 ?
print(train.var38.value_counts().head(15))

train.var38[train['var38'] != 117310.979016494].mean()

# what if we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()

# Look at the distribution
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);


















# Add PCA components as features

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X = train.copy()
X = X.drop('ID', axis=1)
X = X.drop('TARGET', axis=1)
X = X.replace(-999999,2)
X.info(verbose=True)


fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(X.iloc[:,1], X.iloc[:, 2])
ax.scatter(X.iloc[:,4], X.iloc[:, 3])

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    # compute the covariance matrix
    X = np.matrix(X)
    m = X.shape[0]
    cov = (X.T * X) / m 
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    return U, S, V

U, S, V = pca(X)
U, S, V






X = train.copy()
X_normalized = normalize(X, axis=0)
pca = PCA(n_components=300, svd_solver='full')
X_pca = pca.fit_transform(X_normalized)



pca_sigma = pca.components_
pca_eigen = pca.explained_variance_
pca_eigen_ratio = pca.explained_variance_ratio_

X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale




































