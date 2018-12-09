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

```ruby
# fill missing age with median age for each title
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
```

```ruby
# binning
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,    
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4,
```
