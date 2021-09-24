#-------------------------------------------------------------------------
# AUTHOR: Seungyun Lee
# FILENAME: knn.py
# SPECIFICATION: A Python program that finds the error rate of a given data with 1NN.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

def evaluateClassLabel(label):
    if label == '-':
        return 0
    else:
        return 1

def generateX(db, test_point):
    x = []
    for point in db:
        if point != test_point:
            x.append([float(point[0]), float(point[1])])
    return x

def generateY(db, test_point):
    y = []
    for point in db:
        if point != test_point:
            if point[2] == '-':
                y.append(0)
            else:
                y.append(1)
    return y

db = []
num_of_wrong_predictions = 0
total_num_of_predictions = 0;

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)



#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = generateX(db, instance)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    Y = generateY(db, instance)

    #store the test sample of this iteration in the vector testSample
    # testSample =
    #--> add your Python code here
    testSample = []
    testSample.append([float(instance[0]), float(instance[1])])

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict(testSample)[0]
    print(class_predicted)
    total_num_of_predictions = total_num_of_predictions + 1

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != evaluateClassLabel(instance[2]):
        num_of_wrong_predictions = num_of_wrong_predictions + 1


#print the error rate
#--> add your Python code here
print("Error rate with 1NN =", num_of_wrong_predictions/total_num_of_predictions)






