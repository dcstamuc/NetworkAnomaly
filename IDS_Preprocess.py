
import csv
import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np

file_name = "IDS2017_1M"


def test():
    features = []
    labels = []

    with open("../dataset/IDS/dataset_IDS2017_1M_0.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader, delimiter=',')

        i = 0
        for row in reader:
            print(len(row))
            i += 1

            if i == 10:
                break


def load_dataset():
    features = []
    labels = []

    with open("../dataset/IDS/" + file_name + ".csv", "r") as csvReader:
        reader = csv.reader(csvReader, delimiter=',')

        i = 0
        for row in reader:
            # print(row)
            features.append(row[7:-1])
            labels.append(row[-1])

            if len(row) != 85:
                print("Different length: " + str(len(row)))
                print("Its index: " + str(i))

            i += 1
            # if i == 10:
            #     break

    print(len(features[0]))
    print("Total number of records: " + str(len(features)))

    floated = []
    final_labels = []
    numOfWrong = 0

    for i in range(len(features)):
        temp_feature = []
        flag = False
        for j in range(len(features[i])):
            if features[i][j] == "NaN" or features[i][j] == "Infinity":
                flag = True
            else:
                temp_feature.append(float(features[i][j]))

        if flag == True:
            numOfWrong += 1
            del temp_feature
        else:
            floated.append(temp_feature)
            final_labels.append(labels[i])

    print("Data converting is done")
    print("Number of wrong record: " + str(numOfWrong))

    scaler = MinMaxScaler()
    # features = np.array(features)
    normalized = scaler.fit_transform(floated)

    print(normalized[0])
    print(final_labels[0])

    print("All features normalized")
    print("Number of normalized records: " + str(len(normalized)))
    print("Number of the labels: " + str(len(final_labels)))

    return normalized, final_labels


def write_dataset(features, labels):

    dataset = []
    labelset = []

    src = 0
    dst = 100000
    for i in range(10):
        print(src, dst)

        if i >= 9:
            dataset.append(features[src:])
            labelset.append(labels[src:])
        else:
            dataset.append(features[src:dst])
            labelset.append(labels[src:dst])
            src = dst
            dst = dst + 100000

        print(len(dataset[i]))
        print("----------------------")

    dataset_list = []

    for i in range(len(dataset)):
        temp = dataset[i].tolist()
        dataset_list.append(temp)

    print(dataset_list[0][0])
    print(type(dataset_list[0][0]))
    print(type([labelset[0][0]]))

    print("Number of dataset: " + str(len(dataset)))

    for i in range(len(dataset_list)):
        with open("../dataset/IDS/dataset_" + file_name + "_" + str(i) + ".csv", "w") as csvWriter:
            writer = csv.writer(csvWriter, delimiter=',')

            for j in range(len(dataset_list[i])):
                writer.writerow(dataset_list[i][j] + [labelset[i][j]])


if __name__ == "__main__":
    start = datetime.datetime.now()

    # test()
    features, labels = load_dataset()
    write_dataset(features, labels)

    end = datetime.datetime.now()

    print("Running time: " + str(end - start))




