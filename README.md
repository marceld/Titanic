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

fill missing age with median age for each title
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
