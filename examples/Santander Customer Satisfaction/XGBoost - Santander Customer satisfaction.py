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
print("\n======================================================================")
print("\n== VAR3 == \nPrinting Top-10 most common values from %i values\n" % len(train.var3.unique()) )
print_pretty_table(df[:10].transpose())

train.var3.value_counts(dropna=False)[:5].plot('bar'); 
plt.pause(0.01)
