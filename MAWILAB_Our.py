
"""
Created by Taejoon Kim
Thursday. Oct. 4
Generating images with 24 bits encoding
"""

import cv2
import csv
import numpy as np
import os

dir_name = "MAWILAB_Image_"


def load_dataset(step):
    print("======================= loading dataset =======================")

    file_name = "final_mawilab_dataset_"

    dataset = []
    dataset_label = []

    with open("../dataset/MAWILAB/" + file_name + str(step) + ".csv", "rU") as csv_reader:
        reader = csv.reader(csv_reader, delimiter=',')

        i = 0
        for row in reader:
            dataset.append(row[0:-1])
            dataset_label.append(row[-1])

            i += 1
            # if i == 3:
            #     break

    print(len(dataset[0]))
    print(dataset[0])
    print(dataset[0][18])
    print(dataset[0][19])
    print(dataset_label[0])

    return dataset, dataset_label


def encoding(dataset):
    print("======================= encoding =======================")

    bit_size = 8

    encoded = []

    for i in range(len(dataset)):
        temp = []

        for j in range(len(dataset[i])):
            if float(dataset[i][j]) == 1.0:
                temp.append(pow(2, bit_size) - 1)
            elif float(dataset[i][j]) == 0.0:
                temp.append(float(dataset[i][j]) * pow(2, bit_size))
            else:
                temp.append((float(dataset[i][j])) * (pow(2, bit_size) - 1))

        encoded.append(temp)

    print(len(encoded))
    print(len(encoded[0]))
    print(encoded[0])

    return encoded


def padding(encoded):
    print("======================= padding =======================")

    padding_len = 25 - len(encoded[0])

    padded = []

    for i in range(len(encoded)):
        temp = encoded[i]

        for j in range(padding_len):
            temp.append(0.0)

        padded.append(temp)


    print(len(padded))
    print(len(padded[0]))
    print(padded[0])

    return padded


def generating(encoded, label, step):
    print("======================= generating images =======================")

    for i in range(len(encoded)):
        data = np.zeros([5, 5, 3], dtype=np.uint8)

        imageIndex = 0

        for j in range(5):
            for k in range(5):
                data[j][k] = [encoded[i][imageIndex], encoded[i][imageIndex], encoded[i][imageIndex]]
                imageIndex += 1

        image_dir = dir_name + str(step)

        if not os.path.exists("../results/MAWILAB_Our/" + image_dir):
            os.mkdir("../results/MAWILAB_Our/" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("../results/MAWILAB_Our/" + image_dir + "/malicious"):
            os.mkdir("../results/MAWILAB_Our/" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("../results/MAWILAB_Our/" + image_dir + "/normal"):
            os.mkdir("../results/MAWILAB_Our/" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if label[i] == "anomaly":
            cv2.imwrite("../results/MAWILAB_Our/" + image_dir + "/malicious" + "/record" + str(i) + ".jpg", data)
        elif label[i] == "normal":
            cv2.imwrite("../results/MAWILAB_Our/" + image_dir + "/normal" + "/record" + str(i) + ".jpg", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


if __name__ == "__main__":
    for i in range(4):
        dataset, label = load_dataset(i)
        encoded = encoding(dataset)
        padded = padding(encoded)
        generating(padded, label, i)





