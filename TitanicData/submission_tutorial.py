# python 3
import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('csv/train.csv', 'rU')) 
header = csv_file_object.__next__() 

data=[]
for row in csv_file_object:
    data.append(row)         
data = np.array(data)

# Be aware that each item is currently a string in this format

# turn the Pclass (3rd column) variable into ints
data[0::,2].astype(np.int)


# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

# number_passengers = np.size(data[0::,1].astype(np.float))
# number_survived = np.sum(data[0::,1].astype(np.float))
# proportion_survivors = number_survived / number_passengers


# women_only_stats = data[0::,4] == "female" 
# men_only_stats = data[0::,4] != "female" 
# women_onboard = data[women_only_stats,1].astype(np.float)     
# men_onboard = data[men_only_stats,1].astype(np.float)


# proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
# proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 

# print ('Proportion of women who survived is %s' % proportion_women_survived)
# print ('Proportion of men who survived is %s' % proportion_men_survived)


test_file_object = csv.reader(open('csv/test.csv', 'rU')) 
header = test_file_object.next()

prediction_file = open("genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)