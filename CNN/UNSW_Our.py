
"""
Created by Taejoon Kim
Thursday. Oct. 4
Generating images with 24 bits encoding
"""

import cv2
import csv
import numpy as np
import os

dir_name = "UNSW_Testing"

def load_dataset():
    print("======================= loading dataset =======================")

    file_name = "UNSW_NB15_testing-set_full_x_attack_cat"

    dataset = []
    dataset_label = []

    with open("../dataset/UNSW/" + file_name + ".csv", "rU") as csv_reader:
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

    bit_size = 24

    encoded = []

    for i in range(len(dataset)):
        temp = []

        for j in range(len(dataset[i])):
            if float(dataset[i][j]) == 1.0:
                temp.append(pow(2, bit_size) - 1)
            elif float(dataset[i][j]) == 0.0:
                temp.append(0)
            else:
                temp.append(float(dataset[i][j]) * (pow(2, bit_size) - 1))

        encoded.append(temp)

    print(len(encoded))
    print(len(encoded[0]))
    print(encoded[0])

    return encoded


def padding(encoded):
    print("======================= padding =======================")

    padding_len = 196 - len(encoded[0])

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


def generating(encoded, label):
    print("======================= generating images =======================")

    for i in range(len(encoded)):
        data = np.zeros([14, 14, 3], dtype=np.uint8)

        imageIndex = 0

        for j in range(14):
            for k in range(14):

                # encoded with 24 bits
                temp_encoded_number = '{0:024b}'.format(int(encoded[i][imageIndex]))

                # each 8 bits stored in each color variable
                R = int(temp_encoded_number[0:8], 2)
                G = int(temp_encoded_number[8:16], 2)
                B = int(temp_encoded_number[16:24], 2)

                data[j][k] = [R, G, B]

                imageIndex += 1

        image_dir = dir_name

        if not os.path.exists("../results/UNSW_Our/" + image_dir):
            os.mkdir("../results/UNSW_Our/" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("../results/UNSW_Our/" + image_dir + "/malicious"):
            os.mkdir("../results/UNSW_Our/" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("../results/UNSW_Our/" + image_dir + "/normal"):
            os.mkdir("../results/UNSW_Our/" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if label[i] == '1':
            cv2.imwrite("../results/UNSW_Our/" + image_dir + "/malicious" + "/record" + str(i) + ".jpg", data)
        elif label[i] == '0':
            cv2.imwrite("../results/UNSW_Our/" + image_dir + "/normal" + "/record" + str(i) + ".jpg", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


if __name__ == "__main__":
    dataset, label = load_dataset()
    encoded = encoding(dataset)
    padded = padding(encoded)
    generating(padded, label)





