#-------------------------------------------------------------------------
# AUTHOR: Seungyun Lee
# FILENAME: decision_tree.py
# SPECIFICATION: Finding the lowest accuracy of each decision tree model
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

age_values = {'Young' : 1, 'Prepresbyopic' : 2, 'Presbyopic' : 3}
spectacle_pres_values = {'Myope' : 1, 'Hypermetrope' : 2}
astigmatism_values = {'Yes' : 1, 'No' : 2}
tear_prod_rate_values = {'Normal' : 1, 'Reduced' : 2}
recommend_lenses_values = {'Yes' : 1, 'No' : 2}

def transformEntry(entry):
    new_entry = []

    new_entry.append(age_values[entry[0]])
    new_entry.append(spectacle_pres_values[entry[1]])
    new_entry.append(astigmatism_values[entry[2]])
    new_entry.append(tear_prod_rate_values[entry[3]])

    return new_entry

def transformData(db):
    x = []
    for entry in db:
        x.append(transformEntry(entry))

    return x

def transformLabel(db):
    y = []

    for entry in db:
        last_index = len(entry) - 1
        y.append(recommend_lenses_values[entry[last_index]])

    return y

def readTestData(test_data_file_name):
    test_data = []
    with open(test_data_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for j, line in enumerate(csv_reader):
            if j > 0:
                test_data.append(line)
    return test_data


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    # print(ds)

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    X = transformData(dbTraining)
    # print(X)

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    Y = transformLabel(dbTraining)
    # print(Y)

    lowest_accuracy = 10000000
    #loop your training and test tasks 10 times here
    for i in range (10):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        dbTest = readTestData('contact_lens_test.csv')
        # print(dbTest)

        test_sample_size = len(dbTest)
        correct_predictions = 0

        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            test_x = []
            test_x.append(transformEntry(data))
            class_predicted = clf.predict(test_x)[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            actual_class = recommend_lenses_values[data[4]]
            if class_predicted == actual_class:
                correct_predictions = correct_predictions + 1

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        if (correct_predictions / test_sample_size) < lowest_accuracy:
            lowest_accuracy = (correct_predictions / test_sample_size)

    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("Final accuracy when training on " + ds + ":", lowest_accuracy)

