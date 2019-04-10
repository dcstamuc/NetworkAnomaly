
import cv2
import csv
import numpy as np
import os
import sys

global dataset_name

def load_dataset(dataset_name):
    print("========================= Loading =========================")

    file_name = dataset_name

    train = []
    train_label = []

    with open("../dataset/IDS/" + file_name + ".csv", "rU") as csvReader:
        reader = csv.reader(csvReader, delimiter=',')

        i = 0
        for row in reader:
            train.append(row[0:-1])
            train_label.append(row[-1])

            i += 1

            if i == 100:
                break

    print("Number of training: " + str(len(train)))
    print("Number of features in a record: " + str(len(train[0])))

    return train, train_label


def one_hot_encoding(train):
    print("========================= One Hot Encoding =========================")

    encoded_train = []

    for i in range(len(train)):
        temp = []

        for j in range(len(train[i])):
            if 0.0 <= float(train[i][j]) < 0.1:
                for k in range(10):
                    if k == 0:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.1 <= float(train[i][j]):
                for k in range(10):
                    if k == 1:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.2 <= float(train[i][j]):
                for k in range(10):
                    if k == 2:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.3 <= float(train[i][j]):
                for k in range(10):
                    if k == 3:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.4 <= float(train[i][j]):
                for k in range(10):
                    if k == 4:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.5 <= float(train[i][j]):
                for k in range(10):
                    if k == 5:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.6 <= float(train[i][j]):
                for k in range(10):
                    if k == 6:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.7 <= float(train[i][j]):
                for k in range(10):
                    if k == 7:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.8 <= float(train[i][j]):
                for k in range(10):
                    if k == 8:
                        temp.append(1)
                    else:
                        temp.append(0)

            elif 0.9 <= float(train[i][j]):
                for k in range(10):
                    if k == 9:
                        temp.append(1)
                    else:
                        temp.append(0)

        encoded_train.append(temp)

    print("Encoding for training is done")
    print("Number of encoded train: " + str(len(encoded_train)))
    print("Number of encoded training features: " + str(len(encoded_train[0])))

    return encoded_train


def encoding(padded_train):
    print("========================= Encoding =========================")

    encoded = []

    for i in range(len(padded_train)):
        one_image = []
        one_pixel = 0

        for j in range(len(padded_train[i])):
            if j % 8 == 0:
                one_pixel += padded_train[i][j] * pow(2, 7)

            elif j % 8 == 1:
                one_pixel += padded_train[i][j] * pow(2, 6)

            elif j % 8 == 2:
                one_pixel += padded_train[i][j] * pow(2, 5)

            elif j % 8 == 3:
                one_pixel += padded_train[i][j] * pow(2, 4)

            elif j % 8 == 4:
                one_pixel += padded_train[i][j] * pow(2, 3)

            elif j % 8 == 5:
                one_pixel += padded_train[i][j] * pow(2, 2)

            elif j % 8 == 6:
                one_pixel += padded_train[i][j] * pow(2, 1)

            elif j % 8 == 7:
                one_pixel += padded_train[i][j] * pow(2, 0)

            if j != 0 and (j + 1) % 8 == 0:
                one_image.append(one_pixel)
                one_pixel = 0

        encoded.append(one_image)

    print("Number of encoded record: " + str(len(encoded)))
    print("Number of encoded features: " + str(len(encoded[0])))

    return encoded


def padding(encoded_train):
    print("========================= Padding =========================")
    padded_train = []

    for i in range(len(encoded_train)):
        pad_length = 800 - len(encoded_train[0])

        temp = []
        for j in range(len(encoded_train[i])):
            temp.append(encoded_train[i][j])

        for j in range(pad_length):
            temp.append(0)

        padded_train.append(temp)

    print("Number of padded train: " + str(len(padded_train)))
    print("Number of features after padding: " + str(len(padded_train[0])))

    return padded_train


def generateImages(encoded_train, train_label):
    print("========================= Generating =========================")

    for i in range(len(encoded_train)):
        data = np.zeros([10, 10, 3], dtype=np.uint8)

        imageIndex = 0

        for j in range(10):
            for k in range(10):
                data[j][k] = [encoded_train[i][imageIndex], encoded_train[i][imageIndex], encoded_train[i][imageIndex]]
                imageIndex += 1

        # print(data)

        image_dir = "IDS_Grey_" + dataset_name

        if not os.path.exists("../generatedImages/" + image_dir):
            os.mkdir("../generatedImages/" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("../generatedImages/" + image_dir + "/malicious"):
            os.mkdir("../generatedImages/" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("../generatedImages/" + image_dir + "/normal"):
            os.mkdir("../generatedImages/" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if train_label[i] != "BENIGN":
            cv2.imwrite("../generatedImages/" + image_dir + "/malicious" + "/record" + str(i) + ".jpg", data)
        elif train_label[i] == "BENIGN":
            cv2.imwrite("../generatedImages/" + image_dir + "/normal" + "/record" + str(i) + ".jpg", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


if __name__ == "__main__":
    dataset_name = sys.argv[1]

    train, label = load_dataset(dataset_name)

    one_hot_encoded = one_hot_encoding(train)

    padded = padding(one_hot_encoded)

    encoded = encoding(padded)

    generateImages(encoded, label)




