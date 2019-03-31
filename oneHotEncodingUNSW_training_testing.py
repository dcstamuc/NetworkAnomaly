
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file_name1 = "NB15_training-set"
file_name2 = "NB15_testing-set"


def loadDataset():
    pre_dataset = []
    with open("./ProcessedDataset/pre_" + file_name1 + ".csv", "rU") as file:
        reader = csv.reader(file, delimiter=',')

        x = 0
        for row in reader:
            # print(row)
            pre_dataset.append(row)

            x += 1
            # if x == 30:
            #     break
        print("Number of training set: " + str(x))

    with open("./ProcessedDataset/pre_" + file_name2 + ".csv", "rU") as file:
        reader = csv.reader(file, delimiter=',')

        y = 0
        for row in reader:
            pre_dataset.append(row)

            y += 1

        print("Number of testing set: " + str(y))

    return pre_dataset


def separateData(pre_dataset):
    categorical_data = []
    numerical_data = []

    for i in range(len(pre_dataset)):
        temp = []
        temp.append(pre_dataset[i][0])
        temp.append(pre_dataset[i][1])
        temp.append(pre_dataset[i][5])

        categorical_data.append(temp)

        temp2 = []
        for j in range(len(pre_dataset[i][2:5])):
            temp2.append(pre_dataset[i][j + 2])

        for j in range(len(pre_dataset[i][6:])):
            temp2.append(pre_dataset[i][j + 6])

        numerical_data.append(temp2)

    # print(len(categorical_data))
    # print(len(numerical_data))
    #
    # print(len(categorical_data[0]))
    # print(len(numerical_data[0]))
    #
    # print(categorical_data)
    # print(numerical_data)

    return categorical_data, numerical_data


def oneHotEncoding(categories, numerics):
    categorical_data = categories
    numerical_data = numerics

    category1 = {}
    category2 = {}
    category3 = {}

    for i in range(len(categorical_data)):
        category1[categorical_data[i][0]] = 0
        category2[categorical_data[i][1]] = 0
        category3[categorical_data[i][2]] = 0

    encoded_category1 = []
    encoded_category2 = []
    encoded_category3 = []

    """
    one hot encoding for the first categorical variable
    """
    for i in range(len(categorical_data)):
        temp = []

        for cate1 in category1:
            if categorical_data[i][0] == cate1:
                temp.append(1)
            else:
                temp.append(0)

        encoded_category1.append(temp)

    """
    one hot encoding for the second categorical variable
    """
    for i in range(len(categorical_data)):
        temp = []

        for cate2 in category2:
            if categorical_data[i][1] == cate2:
                temp.append(1)
            else:
                temp.append(0)

        encoded_category2.append(temp)

    """
    one hot encoding for the third categorical variable
    """
    for i in range(len(categorical_data)):
        temp = []

        for cate3 in category3:
            if categorical_data[i][2] == cate3:
                temp.append(1)
            else:
                temp.append(0)

        encoded_category3.append(temp)

    # print("-----------------------------------------------------------")
    # print(categorical_data)
    # print("===========================================================")
    # print(encoded_category1)
    # print(encoded_category2)
    # print(encoded_category3)
    # print("===========================================================")
    # print(len(encoded_category1))

    numerical_data = np.array(numerical_data)

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(numerical_data[:, 0:-1])

    """
    merge one hot encoded variables and numerical variables
    """

    dataset = []
    for i in range(len(normalized)):
        temp = []
        temp = encoded_category1[i]
        for j in range(len(encoded_category2[i])):
            temp.append(encoded_category2[i][j])

        for j in range(len(numerical_data[i][0:3])):
            temp.append(normalized[i][j])

        for j in range(len(encoded_category3[i])):
            temp.append(encoded_category3[i][j])

        for j in range(len(numerical_data[i][3:-1])):
            temp.append(normalized[i][j + 3])

        temp.append(numerical_data[i][-1])

        dataset.append(temp)

    return dataset


def saveDataset(data):
    dataset = data
    training_set = data[0:82332]
    testing_set = data[82332:]

    with open("./FinalDataset/final_" + file_name1 + ".csv", "w") as file:
        writer = csv.writer(file)

        x = 0
        for i in range(len(training_set)):
            writer.writerow(training_set[i])
            x += 1

            if i % 10000 == 0 and i != 0:
                print("%ith rows are saved" % i)

        print("After creating training dataset: " + str(x))

    with open("./FinalDataset/final_" + file_name2 + ".csv", "w") as file:
        writer = csv.writer(file)

        y = 0
        for i in range(len(testing_set)):
            writer.writerow(testing_set[i])
            y += 1

            if i % 10000 == 0 and i != 0:
                print("%ith rows are saved" % i)

        print("After creating testing dataset: " + str(y))


pre_dataset = loadDataset()
categories, numerics = separateData(pre_dataset)
dataset = oneHotEncoding(categories, numerics)
saveDataset(dataset)





