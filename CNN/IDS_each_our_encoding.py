
import cv2
import csv
import numpy as np
import os

train_name = "final_IDS2017_M2F_1M_Comma_"


def load_dataset(step):
    print("========================= Loading =========================")

    train = []
    train_label = []

    with open("../dataset/IDS/" + train_name + str(step) + ".csv", "rU") as csvReader:
        reader = csv.reader(csvReader, delimiter=',')

        i = 0
        for row in reader:
            train.append(row[0:-1])
            train_label.append(row[-1])

            i += 1

            if i == 10:
                break

    print("Number of training: " + str(len(train)))
    print("Number of features in a record: " + str(len(train[0])))

    return train, train_label


def padding(train):
    print("========================= Padding =========================")

    padded_train = []

    for i in range(len(train)):
        pad_length = 81 - len(train[i])
        temp = []

        for j in range(len(train[i])):
            temp.append(train[i][j])

        for k in range(pad_length):
            temp.append(0)

        padded_train.append(temp)

    print("Number of training: " + str(len(train)))
    print("Number of padded training: " + str(len(padded_train)))
    print("Number of training features after padding: " + str(len(padded_train[0])))

    return padded_train


def encoding(train):
    print("========================= Encoding =========================")

    bit_size = 8

    encoded_train = []

    print(len(train))
    print(len(train[0]))

    for i in range(len(train)):
        temp = []

        for j in range(len(train[i])):
            if float(train[i][j]) == 1.0:
                temp.append(pow(2, bit_size) - 1)
            elif float(train[i][j]) == 0.0:
                temp.append(float(train[i][j]) * pow(2, bit_size))
            else:
                temp.append(float(train[i][j]) * pow(2, bit_size) - 1)
        print(temp)

        encoded_train.append(temp)

    print("Encoding for training is done")

    print("Number of encoded train: " + str(len(encoded_train)))
    print("Number of encoded training features in a record: " + str(len(encoded_train[0])))

    return encoded_train


def generateImages(encoded_train, train_label, step):
    print("========================= Generating =========================")

    for i in range(len(encoded_train)):
        data = np.zeros([9, 9, 3], dtype=np.uint8)

        imageIndex = 0

        for j in range(9):
            for k in range(9):
                data[j][k] = [encoded_train[i][imageIndex], encoded_train[i][imageIndex], encoded_train[i][imageIndex]]
                imageIndex += 1

        image_dir = "IDS_Train" + "_" + str(step)

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
    for i in range(1):
        train, train_label = load_dataset(i)
        encoded_train = encoding(train)
        padded_train = padding(encoded_train)
        generateImages(padded_train, train_label, i)







