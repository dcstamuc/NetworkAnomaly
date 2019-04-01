
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_dataset():
    dataset = []
    numerical_data = []
    categorical_data = []
    labels = []
    num_training = 0
    num_testing = 0

    with open("../dataset/UNSW/UNSW_NB15_training-set.csv", "rU") as csvReader:
        reader = csv.reader(csvReader, delimiter=',')

        i = 0
        for row in reader:
            temp = []
            categorical_data.append(row[2])

            temp.append(row[5])
            temp.append(row[6])
            temp.append(row[7])
            temp.append(row[8])
            numerical_data.append(temp)

            labels.append(row[-2])
            i += 1
            num_training += 1
            # if i == 10:
            #     break

    print("Number of numerical features: " + str(len(numerical_data[0])))
    print("Number of categorical features: " + str(len(categorical_data[0])))

    with open("../dataset/UNSW/UNSW_NB15_testing-set.csv", "rU") as csvReader:
        reader = csv.reader(csvReader, delimiter=',')

        i = 0
        for row in reader:
            temp = []
            categorical_data.append(row[2])

            temp.append(row[5])
            temp.append(row[6])
            temp.append(row[7])
            temp.append(row[8])
            numerical_data.append(temp)
            labels.append(row[-2])
            i += 1
            num_testing += 1

    print("Number of training set: " + str(num_training))
    print("Number of testing set: " + str(num_testing))
    print("Number of whole dataset: " + str(len(numerical_data)))

    return numerical_data, categorical_data, num_training, num_testing, labels


def normalize_numerical(numerical_data):
    numerical_data = np.array(numerical_data)

    scaler = MinMaxScaler()

    print(numerical_data)
    print(len(numerical_data))
    normalized = scaler.fit_transform(numerical_data[:, :])
    print("Numerical values are normalized")

    return normalized.tolist()


def one_hot_encoding(categorical_data):
    categori1 = {}

    for i in range(len(categorical_data)):
        categori1[categorical_data[i][0]] = 0

    one_hot_encoded0 = []

    for i in range(len(categorical_data)):
        temp = []
        for item in categori1:
            if categorical_data[i][0] == item:
                temp.append(1)
            else:
                temp.append(0)

        one_hot_encoded0.append(temp)

    print("Number of 1st category: " + str(len(one_hot_encoded0)))

    return one_hot_encoded0


def merge_data(normalized, one_hot0):

    print(len(normalized[0]))

    for i in range(len(normalized)):
        for j in range(len(one_hot0[i])):
            normalized[i].append(one_hot0[i][j])

    print("Number of features after one hot encoding and normalization: " + str(len(normalized[0])))
    print("Number of whole records: " + str(len(normalized)))

    return normalized


def attach_labels(normalized, labels):
    for i in range(len(normalized)):
        if labels[i] == "Normal":
            normalized[i].append(0)
        else:
            normalized[i].append(1)

    return normalized


def separate_training_testing(normalized, num_training, num_testing):
    training_set = normalized[0:num_training]
    testing_set = normalized[num_training:]

    print("Number of training set: " + str(len(training_set)))
    print("Number of testing set: " + str(len(testing_set)))

    with open("./UNSW_partial_NB15_training-set.csv", "w") as csvWriter:
        writer = csv.writer(csvWriter)

        for i in range(len(training_set)):
            writer.writerow(training_set[i])

    with open("./UNSW_partial_NB15_testing-set.csv", "w") as csvWriter:
        writer = csv.writer(csvWriter)

        for i in range(len(testing_set)):
            writer.writerow(testing_set[i])


numerical_data, categorical_data, num_training, num_testing, labels = load_dataset()

normalized = normalize_numerical(numerical_data)
one_hot0 = one_hot_encoding(categorical_data)

normalized = merge_data(normalized, one_hot0)
normalized_label = attach_labels(normalized, labels)
separate_training_testing(normalized_label, num_training, num_testing)








