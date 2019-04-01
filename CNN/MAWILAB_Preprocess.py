
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_dataset():
    print("===================== loading dataset =====================")

    dataset = []
    dataset_cnt = []

    for file_cnt in range(4):
        with open("../dataset/MAWILAB/20180511_mawilab_flow-5sec-" + str(file_cnt) + ".csv", "r") as csv_reader:
            reader = csv.reader(csv_reader, delimiter=',')

            i = 0
            for row in reader:
                # print(row)

                if i == 0:
                    i += 1
                    continue

                temp = []
                if(row[23] != "unsure"):
                    temp.append(row[7])
                    temp.append(row[4])
                    temp.append(row[5])
                    temp.append(row[6])
                    temp.append(row[9])
                    temp.append(row[23]) # label which is normal, anomaly, and unsure, in this case, unsure is excepted
                    dataset.append(temp)
                    i += 1

                # if i == 100:
                #     break

            dataset_cnt.append(i - 1)

    print("Number of record: " + str(len(dataset)))
    print(dataset[0])
    print(dataset[1])
    print(dataset_cnt)

    return dataset, dataset_cnt


def one_hot_encoding(dataset):
    print("===================== one hot encoding =====================")
    encoded_items = []
    encoded = []

    category = {}

    for i in range(len(dataset)):
        category[dataset[i][0]] = 0

    print(category)
    print("Number of cateogrical item: " + str(len(category)))

    for i in range(len(dataset)):
        temp = []

        for item in category:
            if item == dataset[i][0]:
                temp.append(1)
            else:
                temp.append(0)

        encoded_items.append(temp)

    for i in range(len(dataset)):
        temp = []

        for j in range(len(encoded_items[i])):
            temp.append(encoded_items[i][j])

        for value in dataset[i][1:]:
            temp.append(value)

        encoded.append(temp)

    print("Number of encoded record: " + str(len(encoded)))
    print("Number of each encoded record: " + str(len(encoded[0])))
    print(encoded[0])

    return encoded


def normalizing(encoded):
    print("===================== min max normalizing =====================")

    encoded = np.array(encoded)

    scaler = MinMaxScaler()

    normalizing = scaler.fit_transform(encoded[:, 0:-1]) # except label, for whole row and column index 0 to -1

    normalized = normalizing.tolist()

    for i in range(len(encoded)):
        normalized[i].append(encoded[i][-1]) # put label at the end of each record

    print("Number of noramlized record: " + str(len(normalized)))
    print("Number of each noramlized record: " + str(len(normalized[0])))
    print(normalized[0])

    return normalized


def separate_dataset(normalized, dataset_cnt):
    print("===================== separating dataset =====================")

    train_index = dataset_cnt[0]
    test1_index = train_index + dataset_cnt[1]
    test2_index = test1_index + dataset_cnt[2]
    test3_index = test2_index + dataset_cnt[3]

    print("Dataset index: " + str(train_index) + " " + str(test1_index) + " " + str(test2_index) + " " + str(test3_index))

    training = normalized[0:train_index]
    testing1 = normalized[train_index:test1_index]
    testing2 = normalized[test1_index:test2_index]
    testing3 = normalized[test2_index:]

    print("Number of training set: " + str(len(training)))
    print("Number of testing set 1: " + str(len(testing1)))
    print("Number of testing set 2: " + str(len(testing2)))
    print("Number of testing set 3: " + str(len(testing3)))

    dataset = []
    dataset.append(training)
    dataset.append(testing1)
    dataset.append(testing2)
    dataset.append(testing3)

    return dataset


def write_dataset(dataset):

    for i in range(len(dataset)):
        with open("../dataset/MAWILAB/final_mawilab_dataset_" + str(i) + ".csv", "w") as csv_writer:
            writer = csv.writer(csv_writer)

            for j in range(len(dataset[i])):
                writer.writerow(dataset[i][j])


dataset, dataset_cnt = load_dataset()
encoded = one_hot_encoding(dataset)
normalized = normalizing(encoded)
dataset = separate_dataset(normalized, dataset_cnt)
write_dataset(dataset)








