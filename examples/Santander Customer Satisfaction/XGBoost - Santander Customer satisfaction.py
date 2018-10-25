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

train.dtypes
train.head()

#==============================================================================
#                Exploratory Data Analysis - EDA
#==============================================================================
def print_pretty_table(data):
    print(tabulate(data, headers='keys', tablefmt='psql'))

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
print("\n== VAR3 == \nPrinting Top-10 most common values from %i values\n" % len(train.var3.unique()) )
# print_pretty_table(df[:5].transpose())
print_pretty_table(df[:5])

train.var3.value_counts(dropna=False)[:5].plot('bar'); 
plt.pause(0.01)













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




































