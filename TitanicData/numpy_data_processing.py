import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier


def dataprocess(df):
	# convert gender and embarked strings to ints
	df['Gender'] = df.Sex.map( {'female': 0, 'male': 1} ).astype(int)
	df.loc[ (df.Embarked.isnull()), 'Embarked'] = 'S' # S is median (only two null values)
	df.loc[ (df.Fare.isnull()), 'Fare'] = df.Fare.median()
	df['EmbarkedNumeric'] = df.Embarked.map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

	# fill in median ages for missing ages by gender and pclass
	df['AgeFill'] = df.Age
	median_ages = np.zeros((2,3))
	for i in range(0, 2):
	    for j in range(0, 3):
	        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

	for i in range(0,2):
	    for j in range(0,3):
	        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

	df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
	df['FamilySize'] = df['SibSp'] + df['Parch']
	df['Age*Class'] = df.AgeFill * df.Pclass

	df = df.drop(['PassengerId' ,'Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

	# convert pandas df to numpy array
	return (df.values)




train_df = pd.read_csv('csv/train.csv', header=0)
train_data = dataprocess(train_df)

test_df = pd.read_csv('csv/test.csv', header=0)
test_data = dataprocess(test_df)


# for zipping row ids in final output
test_file_object = csv.reader(open('csv/test.csv', 'rU'))
header = test_file_object.__next__()
ids = []
for row in test_file_object:
    ids.append(row[0])


print ('Training')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::,1::], train_data[0::, 0])

print ('Predicting')
output = forest.predict(test_data)

output_file = csv.writer(open("csv/randforest.csv", "w"))
output_file.writerow(["PassengerId","Survived"])
# output_file.writerows(zip(test_df.values[::,0], output))
output_file.writerows(zip(ids, output.astype(int)))








