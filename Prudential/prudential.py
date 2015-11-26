import json
import os
import pandas as pd
import numpy as np
import xlsxwriter
import csv as csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import neighbors, linear_model, cross_validation

pd.set_option("display.max_rows",10)
pd.set_option("display.max_columns",100)


def round_array(a, decimals):
     return np.around(a-10**(-(decimals+5)), decimals=decimals)


def print_full(x):
	pd.set_option('display.max_rows', len(x))
	print(x)
	pd.reset_option('display.max_rows')


def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


# input model, X training data, y training data, X test data, and ids
# returns y_test and creates output file with predictions
def create_output(model, X_train, y_train, X_test, ids):
	print ('Generating predictions for output file')
	predictions = model.fit(X_train, y_train).predict(X_test)
	predictions = round_array(predictions,0)


	f = open("output.csv", "w")
	writer = csv.writer(f)
	writer.writerow(["Id","Response"])
	writer.writerows(zip(ids, predictions.astype(int)))
	f.close()

	print ('Complete. Output file saved.')


def dataprocess(df):

	# df.info(verbose = True, null_counts = True)

	# fill in null values
	df.loc[ (df.Employment_Info_1.isnull()), 'Employment_Info_1'] = df.Employment_Info_1.mean()
	df.loc[ (df.Employment_Info_4.isnull()), 'Employment_Info_4'] = df.Employment_Info_4.mean()
	df.loc[ (df.Employment_Info_6.isnull()), 'Employment_Info_6'] = df.Employment_Info_6.mean()

	df.loc[ (df.Insurance_History_5.isnull()), 'Insurance_History_5'] = df.Insurance_History_5.mean()
	
	df.loc[ (df.Family_Hist_2.isnull()), 'Family_Hist_2'] = df.Family_Hist_2.mean()
	df.loc[ (df.Family_Hist_3.isnull()), 'Family_Hist_3'] = df.Family_Hist_3.mean()	
	df.loc[ (df.Family_Hist_4.isnull()), 'Family_Hist_4'] = df.Family_Hist_4.mean()
	df.loc[ (df.Family_Hist_5.isnull()), 'Family_Hist_5'] = df.Family_Hist_5.mean()		

	df.loc[ (df.Medical_History_1.isnull()), 'Medical_History_1'] = df.Medical_History_1.mean()
	df.loc[ (df.Medical_History_10.isnull()), 'Medical_History_10'] = df.Medical_History_10.mean()
	df.loc[ (df.Medical_History_15.isnull()), 'Medical_History_15'] = df.Medical_History_15.mean()
	df.loc[ (df.Medical_History_24.isnull()), 'Medical_History_24'] = df.Medical_History_24.mean()
	df.loc[ (df.Medical_History_32.isnull()), 'Medical_History_32'] = df.Medical_History_32.mean()


	# new variables
	# df['Age*BMI'] = df.Ins_Age * df.BMI

	# create dummy variable (truth value for Product_Info_2)
	# Product_Info_2_dummies = pd.get_dummies(df['Product_Info_2'], prefix='Product_Info_2')
	# df = pd.concat([df, Product_Info_2_dummies], axis=1)

	# df['Product_Info_2_Numeric'] = df.Product_Info_2.map( 
	# 	{'C1': 10, 'A1': 0, 'C3': 12, 'A3': 2, 'D3': 16, 
	# 	'A8': 7, 'D4': 17, 'A4': 3, 'C2': 11, 'A2': 1, 'A7': 6, 
	# 	'A6': 5, 'B1': 8, 'B2': 9, 'D1': 14, 'A5': 4, 'E1': 18, 
	# 	'C4': 13, 'D2': 15} ).astype(int)

	df = df.drop(['Id', 'Product_Info_2'], axis=1) 

	return df



train_df = pd.read_csv('train.csv', header=0)
train_data = dataprocess(train_df)

test_df = pd.read_csv('test.csv', header=0)
test_data = dataprocess(test_df)
ids = test_df.Id.values # for zipping output


n_samples = len(train_data)
train_cols = train_data.columns
train_cols = train_cols.drop(['Response'])


# train data has 75% of records
# X_train = train_data[train_cols].values[:.75 * n_samples]
# y_train = train_data['Response'].values[:.75 * n_samples]
# test data has 25% of records
# X_test = train_data[train_cols].values[.75 * n_samples:]
# y_test = train_data['Response'].values[.75 * n_samples:]

# split training data for cross validation (same as above)
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(
# 	train_data[train_cols].values, train_data['Response'].values,
# 	test_size=0.25, random_state=0)

# classifier fit (X,y).predict(t)
# X = train_data[train_cols].values (data in array format)
# y = train_data['Response'].values (target in array format)
# t = test_data

print ('Training knn')
knn = neighbors.KNeighborsClassifier()
knn_scores = cross_validation.cross_val_score(
	knn, train_data[train_cols].values, train_data['Response'].values, cv=5)
# knn_score = knn.fit(X_train, y_train).score(X_test, y_test)

print ('Training logistic')
log_reg = linear_model.LogisticRegression()
log_reg_scores = cross_validation.cross_val_score(
	log_reg, train_data[train_cols].values, train_data['Response'].values, cv=5)
# logistic_score = log_reg.fit(X_train, y_train).score(X_test, y_test)

print ('Training svc' + '\n')
svc = OneVsRestClassifier(LinearSVC(random_state=0))
svc_scores = cross_validation.cross_val_score(
	svc, train_data[train_cols].values, train_data['Response'].values, cv=5)
# svc_score = svc.fit(X_train, y_train).score(X_test, y_test)

print ('KNN score: ' + str(knn_scores))
print ('LogisticRegression score: ' + str(log_reg_scores))
print ('SVC score: ' + str(svc_scores) + '\n')


# reset training and test values
X_train = train_data[train_cols].values
y_train = train_data['Response'].values
X_test = test_data.values
# y_test = []








