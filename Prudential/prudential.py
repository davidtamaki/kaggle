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
# returns y_test and creates output.csv file with predictions
# e.g. create_output(knn, X_train, y_train, X_test, ids)
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
	df['Age+BMI'] = df.Ins_Age + df.BMI
	df['Wt+BMI'] = df.Wt + df.BMI
	df['Ht-BMI'] = df.Ht - df.BMI

	# group weight into quintiles
	weight_quantiles = df.Wt.quantile([0,.2,.4,.6,.8,1]).values
	for i in range(0, 5):
		df['Wt_' + str(i+1)] = (train_df['Wt'] >= weight_quantiles[i]) & (train_df['Wt'] <= weight_quantiles[i+1])
		df['Wt_' + str(i+1)] = df['Wt_' + str(i+1)].astype(int)
	# double counting some values (quantile cannot be greater than 1) need to fix!


	# sum of Medical_Keyword_1 - 48
	df["Sum_Medical_Keyword"] = 0
	for i in range(1,49):
		df["Sum_Medical_Keyword"] = df["Sum_Medical_Keyword"] + df["Medical_Keyword_" + str(i)]


	# group 8-16 together
	df.Sum_Medical_Keyword[df.Sum_Medical_Keyword >= 8] = 8

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




train_df = pd.read_csv('csv/train.csv', header=0)
train_data = dataprocess(train_df)

test_df = pd.read_csv('csv/test.csv', header=0)
test_data = dataprocess(test_df)
ids = test_df.Id.values # for zipping output


n_samples = len(train_data)
train_cols = train_data.columns
train_cols = train_cols.drop(['Response'])

# classifier fit (X,y).predict(t)
# X = train_data[train_cols].values (data in array format)
# y = train_data['Response'].values (target in array format)
# t = test_data

print ('Training knn')
knn = neighbors.KNeighborsClassifier()
knn_scores = cross_validation.cross_val_score(
	knn, train_data[train_cols].values, train_data['Response'].values, cv=3)

print ('Training logistic')
log_reg = linear_model.LogisticRegression()
log_reg_scores = cross_validation.cross_val_score(
	log_reg, train_data[train_cols].values, train_data['Response'].values, cv=3)

print ('Training svc' + '\n')
svc = OneVsRestClassifier(LinearSVC(random_state=0))
svc_scores = cross_validation.cross_val_score(
	svc, train_data[train_cols].values, train_data['Response'].values, cv=3)


print ("KNN score: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))
print ("Logistic Regression score: %0.2f (+/- %0.2f)" % (log_reg_scores.mean(), log_reg_scores.std() * 2))
print ("SVC score: %0.2f (+/- %0.2f) \n" % (svc_scores.mean(), svc_scores.std() * 2))

# set training and test values
X_train = train_data[train_cols].values
y_train = train_data['Response'].values
X_test = test_data.values








