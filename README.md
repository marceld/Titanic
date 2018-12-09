# Machine Learning from Disaster 

## Kaggle competition 

- Predict survival on the Titanic
- Define problem statement
- Collect data
- Exploratory data analysis
- Feature engineering
- Feature selection
- Modelling
- Testing

### Using classifier models from the scikit-learn machine learning library and jupyter notebooks, the best solution has an accuracy of 0.775.

## Code examples 

### Feature engineering

```python
# fill missing age with median age for each title

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
```

```python
# binning

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,    
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4,
```
```python
# filling in missing values

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

plt.show()
```
```python
# classifier models imported from scikit-learn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
```
