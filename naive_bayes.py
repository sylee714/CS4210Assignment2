#-------------------------------------------------------------------------
# AUTHOR: Seungyun Lee
# FILENAME: naive_bayes.py
# SPECIFICATION: Train a naive bayes model on weather data and test with a given test data. Print data entries with more than 0.75 confidence.
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

outlook_values = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_values = {"Hot": 1, "Mild" : 2, "Cool": 3}
humidity_values = {"High" : 1, "Normal": 2}
wind_values = {"Weak": 1, "Strong": 1}
play_tennis_values = {"Yes": 1, "No": 2}


def transformEntry(entry):
    new_entry = []

    new_entry.append(outlook_values[entry[1]])
    new_entry.append(temperature_values[entry[2]])
    new_entry.append(humidity_values[entry[3]])
    new_entry.append(wind_values[entry[4]])

    return new_entry

def transformData(data):
    new_x = []

    for entry in data:
        new_x.append(transformEntry(entry))

    return new_x

def transformLabel(data):
    new_y = []

    for entry in data:
        new_y.append(play_tennis_values[entry[5]])

    return new_y


#reading the training data
#--> add your Python code here
data_training = []

with open('weather_training.csv', 'r') as csv_training_file:
    reader = csv.reader(csv_training_file)
    for i, row in enumerate(reader):
        if i > 0:
            data_training.append(row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
X = transformData(data_training)
print(X)
print('----------------------------------------')

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y = transformLabel(data_training)
print(Y)
print('----------------------------------------')

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
data_test = []
with open('weather_test.csv', 'r') as csv_test_file:
    reader = csv.reader(csv_test_file)
    for i, row in enumerate(reader):
        if i > 0:
            data_test.append(row)

# need to only get features except 'Day'
test_X = transformData(data_test)
print(test_X)
print('----------------------------------------')



#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
predicted = clf.predict_proba(test_X)
print(predicted)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

for i,entry in enumerate(data_test):
    yes_confidence = predicted[i][0]
    no_confidence = predicted[i][1]
    if yes_confidence >= 0.75:
        print(entry[0].ljust(15) + entry[1].ljust(15) + entry[2].ljust(15) + entry[3].ljust(15) +
              entry[4].ljust(15) + "Yes".ljust(15) + str(yes_confidence).ljust(15))

    if no_confidence >= 0.75:
        print(entry[0].ljust(15) + entry[1].ljust(15) + entry[2].ljust(15) + entry[3].ljust(15) +
              entry[4].ljust(15) + "No".ljust(15) + str(yes_confidence).ljust(15))


