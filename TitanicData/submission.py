import pandas
import numpy
from sklearn.linear_model import LogisticRegression
# from sklearn import cross_validation
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import KFold


# 12 processing test set
titanic_test = pandas.read_csv("csv/test.csv")
titanic_train = pandas.read_csv("csv/train.csv")


titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2




# 13 submission to kaggle
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Train the algorithm using all the training data
alg.fit(titanic_train[predictors], titanic_train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    
submission.to_csv("kaggle.csv", index=False)