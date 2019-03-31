
import cv2
import csv
import numpy as np
import os

dataset_name = "dataset_partial_IDS2017_1M_partial_"


def load_dataset(step):
    print("========================= Loading =========================")

    train = []
    train_label = []

    with open("../dataset/IDS/" + dataset_name + str(step) + ".csv", "rU") as csvReader:
        reader = csv.reader(csvReader, delimiter=',')

        i = 0
        for row in reader:
            train.append(row[0:-1])
            train_label.append(row[-1])

            i += 1

    print("Number of dataset: " + str(len(train)))

    print("Number of features in a record: " + str(len(train[0])))

    return train, train_label


def padding(dataset):
    print("========================= Padding =========================")

    padded_dataset = []

    for i in range(len(dataset)):
        pad_length = 9 - len(dataset[i])
        temp = []

        for j in range(len(dataset[i])):
            temp.append(dataset[i][j])

        for k in range(pad_length):
            temp.append(0)

        padded_dataset.append(temp)

    print("Number of dataset: " + str(len(dataset)))
    print("Number of padded training: " + str(len(padded_dataset)))
    print("Number of datset features after padding: " + str(len(padded_dataset[0])))

    return padded_dataset


def encoding(dataset):
    print("========================= Encoding =========================")

    bit_size = 8

    encoded_dataset = []

    print(len(dataset))
    print(len(dataset[0]))

    for i in range(len(dataset)):
        temp = []

        for j in range(len(dataset[i])):
            if float(dataset[i][j]) == 1.0:
                temp.append(pow(2, bit_size) - 1)
            elif float(dataset[i][j]) == 0.0:
                temp.append(float(dataset[i][j]) * pow(2, bit_size))
            else:
                temp.append((float(dataset[i][j]) * (pow(2, bit_size) - 1)))

            print(temp)

        encoded_dataset.append(temp)

    print("Encoding for dataset is done")

    print("Number of encoded dataset: " + str(len(encoded_dataset)))
    print("Number of encoded dataset features in a record: " + str(len(encoded_dataset[0])))

    return encoded_dataset


def generateImages(encoded_train, train_label, step):
    print("========================= Encoding =========================")

    for i in range(len(encoded_train)):
        data = np.zeros([3, 3, 3], dtype=np.uint8)

        imageIndex = 0

        for j in range(3):
            for k in range(3):
                data[j][k] = [encoded_train[i][imageIndex], encoded_train[i][imageIndex], encoded_train[i][imageIndex]]
                imageIndex += 1

        image_dir = "IDS_Our_Train_" + str(step)

        if not os.path.exists("../results/" + image_dir):
            os.mkdir("../results/" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("../results/" + image_dir + "/malicious"):
            os.mkdir("../results/" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("../results/" + image_dir + "/normal"):
            os.mkdir("../results/" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if train_label[i] != "BENIGN":
            cv2.imwrite("../results/" + image_dir + "/malicious" + "/record" + str(i) + ".png", data)
        elif train_label[i] == "BENIGN":
            cv2.imwrite("../results/" + image_dir + "/normal" + "/record" + str(i) + ".png", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


if __name__ == "__main__":
    for i in range(10):
        train, train_label = load_dataset(i)
        padded_train = padding(train)
        encoded_train = encoding(padded_train)
        generateImages(encoded_train, train_label, i)







