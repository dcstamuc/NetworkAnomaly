
"""
Created by Taejoon Kim
Thursday. Oct. 4
Generating images with grey scale encoding
"""

import cv2
import csv
import numpy as np
import os
import sys

def load_dataset(dataset_name):
    print("======================= loading dataset =======================")

    file_name = "Preprocessed_" + dataset_name

    dataset = []
    dataset_label = []

    with open("../dataset/UNSW/" + file_name + ".csv", "rU") as csv_reader:
        reader = csv.reader(csv_reader, delimiter=',')

        i = 0
        for row in reader:
            dataset.append(row[0:-1])
            dataset_label.append(row[-1])

            i += 1
            if i == 10:
                break

    print(len(dataset[0]))
    print(dataset[0])
    print(dataset[0][18])
    print(dataset[0][19])
    print(dataset_label[0])

    return dataset, dataset_label


def one_hot_encoding(dataset):
    print("======================= one hot encoding =======================")

    already_encoded = []

    for i in range(len(dataset)):
        already_encoded.append(dataset[i][38:])

    need_encoding = []

    for i in range(len(dataset)):
        temp = []
        for value in dataset[i][0:38]:
            temp.append(value)
        need_encoding.append(temp)

    print(len(already_encoded))
    print(len(need_encoding))

    numeric_encoded = []

    for i in range(len(need_encoding)):
        temp = []

        for value in need_encoding[i]:
            if 0.0 <= float(value) < 0.1:
                for j in range(10):
                    if j == 0:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.1 <= float(value) < 0.2:
                for j in range(10):
                    if j == 1:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.2 <= float(value) < 0.3:
                for j in range(10):
                    if j == 2:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.3 <= float(value) < 0.4:
                for j in range(10):
                    if j == 3:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.4 <= float(value) < 0.5:
                for j in range(10):
                    if j == 4:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.5 <= float(value) < 0.6:
                for j in range(10):
                    if j == 5:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.6 <= float(value) < 0.7:
                for j in range(10):
                    if j == 6:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.7 <= float(value) < 0.8:
                for j in range(10):
                    if j == 7:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.8 <= float(value) < 0.9:
                for j in range(10):
                    if j == 8:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

            elif 0.9 <= float(value):
                for j in range(10):
                    if j == 9:
                        temp.append(1.0)
                    else:
                        temp.append(0.0)

        numeric_encoded.append(temp)


    encoded = []

    for i in range(len(numeric_encoded)):
        temp = already_encoded[i]

        for j in range(len(numeric_encoded[i])):
            temp.append(numeric_encoded[i][j])

        encoded.append(temp)

    print(len(already_encoded))
    print(len(encoded))
    print(encoded[0])
    print(len(encoded[0]))

    return encoded


def padding(one_hot_encoded):
    print("======================= padding =======================")

    padded = []

    padding_len = 648 - len(one_hot_encoded[0])

    for i in range(len(one_hot_encoded)):
        temp = one_hot_encoded[i]

        for j in range(padding_len):
            temp.append(0)

        padded.append(temp)

    print(len(padded))
    print(len(padded[0]))
    print(padded[0])

    return padded


def encoding(padded):
    print("======================= encoding =======================")

    encoded = []

    for i in range(len(padded)):
        one_image = []
        one_pixel = 0

        for j in range(len(padded[i])):
            if j % 8 == 0:
                one_pixel += float(padded[i][j]) * pow(2, 7)

            elif j % 8 == 1:
                one_pixel += float(padded[i][j]) * pow(2, 6)

            elif j % 8 == 2:
                one_pixel += float(padded[i][j]) * pow(2, 5)

            elif j % 8 == 3:
                one_pixel += float(padded[i][j]) * pow(2, 4)

            elif j % 8 == 4:
                one_pixel += float(padded[i][j]) * pow(2, 3)

            elif j % 8 == 5:
                one_pixel += float(padded[i][j]) * pow(2, 2)

            elif j % 8 == 6:
                one_pixel += float(padded[i][j]) * pow(2, 1)

            elif j % 8 == 7:
                one_pixel += float(padded[i][j]) * pow(2, 0)

            if j != 0 and (j + 1) % 8 == 0:
                one_image.append(one_pixel)
                one_pixel = 0

        encoded.append(one_image)

    print(len(encoded))
    print(len(encoded[0]))
    print(encoded[0])

    return encoded


def generating(encoded, label):
    print("======================= generating images =======================")

    for i in range(len(encoded)):
        data = np.zeros([8, 8, 3], dtype=np.uint8)

        imageIndex = 0

        for j in range(8):
            for k in range(8):
                data[j][k] = [encoded[i][imageIndex], encoded[i][imageIndex], encoded[i][imageIndex]]
                imageIndex += 1

        image_dir = dataset_name

        if not os.path.exists("../generatedImages/UNSW_Grey"):
            os.mkdir("../generatedImages/UNSW_Grey")

        if not os.path.exists("../generatedImages/UNSW_Grey/" + image_dir):
            os.mkdir("../generatedImages/UNSW_Grey/" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("../generatedImages/UNSW_Grey/" + image_dir + "/malicious"):
            os.mkdir("../generatedImages/UNSW_Grey/" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("../generatedImages/UNSW_Grey/" + image_dir + "/normal"):
            os.mkdir("../generatedImages/UNSW_Grey/" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if label[i] == '1':
            cv2.imwrite("../generatedImages/UNSW_Grey/" + image_dir + "/malicious" + "/record" + str(i) + ".jpg", data)
        elif label[i] == '0':
            cv2.imwrite("../generatedImages/UNSW_Grey/" + image_dir + "/normal" + "/record" + str(i) + ".jpg", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


if __name__ == "__main__":
    dataset_name = sys.argv[1]

    dataset, label = load_dataset(dataset_name)
    one_hot_encoded = one_hot_encoding(dataset)
    padded = padding(one_hot_encoded)
    encoded = encoding(padded)
    generating(encoded, label)






