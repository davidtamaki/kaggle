import pandas as pd
import numpy as np
import csv as csv
import statsmodels.api as sm
import pylab as pl
from sklearn.ensemble import RandomForestClassifier


def round_array(a, decimals):
     return np.around(a-10**(-(decimals+5)), decimals=decimals)


def dataprocess(df):
	# convert gender and embarked strings to ints
	df['Gender'] = df.Sex.map( {'female': 0, 'male': 1} ).astype(int)
	df.loc[ (df.Embarked.isnull()), 'Embarked'] = 'S' # S is median (only two null values)
	df.loc[ (df.Fare.isnull()), 'Fare'] = df.Fare.median()

	# fill in median ages for missing ages by gender and pclass
	df['AgeFill'] = df.Age
	median_ages = np.zeros((2,3))
	for i in range(0, 2):
	    for j in range(0, 3):
	        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

	for i in range(0,2):
	    for j in range(0,3):
	        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

	
	# testing input variables
	df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
	df['FamilySize'] = df['SibSp'] + df['Parch']
	df['Age^2'] = df['AgeFill'] ** 2
	df['Age*Class'] = df['AgeFill'] * df['Pclass']
	df['Fare * Class'] = df['Fare'] * df['Pclass']
	df['Age * Fare'] = df['AgeFill'] * df['Fare']


	# create dummy variable (truth value for Pclass and Embarked)
	# Pclass range from [1,2,3] , Embarked from [C,Q,S] (excluding first dummy variable to prevent dummary variable trap) 
	pclass_dummy = pd.get_dummies(df['Pclass'], prefix='Pclass')
	embarked_dummy = pd.get_dummies(df['Embarked'], prefix='Embarked')
	df = df.join([pclass_dummy.ix[:, 'Pclass_2':], embarked_dummy.ix[:, 'Embarked_Q':]])

	df = df.drop(['PassengerId' ,'Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Pclass', 'Embarked'], axis=1) 

	# convert pandas df to numpy array
	return df




train_df = pd.read_csv('csv/train.csv', header=0)
train_data = dataprocess(train_df)

test_df = pd.read_csv('csv/test.csv', header=0)
# for zipping row ids in final output
ids = test_df.PassengerId.values
test_data = dataprocess(test_df)


# train on first column (rows 0::, column 0 -> first column, i.e. 'Survived') 
# based on all other columns (rows 0::, columns 1::)
print ('Training')
train_cols = train_data.columns[1:]
logit = sm.Logit(train_data['Survived'], train_data[train_cols])
result = logit.fit()

print ('Predicting')
predictions = result.predict(test_data)
predictions = round_array(predictions,0)


f = open("csv/logitregression.csv", "w")
writer = csv.writer(f)
writer.writerow(["PassengerId","Survived"])
writer.writerows(zip(ids, predictions.astype(int)))
f.close()










