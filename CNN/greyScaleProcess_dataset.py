
import csv
import numpy as np

def getFeatures():
    testing_list = []
    training_list = []

    with open("UNSW_NB15_testing-set.csv", "rU") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in reader:
            temp = []
            temp.append(row[2])
            temp.append(row[4])
            temp.append(row[1])
            temp.append(row[7])
            temp.append(row[8])
            temp.append(row[3])
            temp.append(row[12])
            temp.append(row[13])
            temp.append(row[5])
            temp.append(row[6])
            temp.append(row[16])
            temp.append(row[17])
            temp.append(row[24])
            temp.append(row[25])
            temp.append(row[26])
            temp.append(row[-1])
            testing_list.append(temp)

    with open("UNSW_NB15_training-set.csv", "rU") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in reader:
            temp = []
            temp.append(row[2])
            temp.append(row[4])
            temp.append(row[1])
            temp.append(row[7])
            temp.append(row[8])
            temp.append(row[3])
            temp.append(row[12])
            temp.append(row[13])
            temp.append(row[5])
            temp.append(row[6])
            temp.append(row[16])
            temp.append(row[17])
            temp.append(row[24])
            temp.append(row[25])
            temp.append(row[26])
            temp.append(row[-1])
            training_list.append(temp)

    with open("./ProcessedDataset/pre_NB15_training-set.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(len(training_list)):
            writer.writerow(training_list[i])

    with open("./ProcessedDataset/pre_NB15_testing-set.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(len(testing_list)):
            writer.writerow(testing_list[i])


getFeatures()



