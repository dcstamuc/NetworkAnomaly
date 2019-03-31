
import csv
import numpy as np
import cv2
import os

file_name = "NB15_1"

dataset = []


def loadDataset():
    with open("./FinalDataset/final_" + file_name + ".csv", "rU") as file:
        reader = csv.reader(file, delimiter=',')

        i = 0
        for row in reader:
            dataset.append(row)

            # i += 1
            # if i == 1:
            #     break

    return dataset


def encodingDataset(data):

    bit_size = 8

    dataset = data

    encoded_dataset = []
    label = []

    for i in range(len(dataset)):
        temp = []

        for j in range(len(dataset[i][0:-1])):
            if float(dataset[i][j]) == 1.0:
                temp.append(pow(2, bit_size - 1))
            else:
                temp.append(float(dataset[i][j]) * pow(2, bit_size - 1))

        # make sure the image size
        if len(temp) != 197:
            padding_size = 197 - len(temp)

            for j in range(padding_size):
                temp.append(0)

        encoded_dataset.append(temp)
        label.append(dataset[i][-1])

        # print(len(encoded_dataset[0]))
        # print(encoded_dataset[i])
        # print(label[i])

    return encoded_dataset, label


def makePictures(data_set, label):
    dataset = data_set
    labels = label

    print(len(dataset))
    print(len(dataset[0]))

    print("======================================= generate images ========================================")
    for i in range(len(dataset)):
        data = np.zeros([14, 14, 3], dtype=np.uint8)

        # imageIndex = 0
        # for j in range(3):
        #     index_count = 0
        #     for k in range(12):
        #         for l in range(12):
        #             # data[j][k] = [dataset[i][imageIndex], dataset[i][imageIndex], dataset[i][imageIndex]]
        #             data[j][k][l] = dataset[i][index_count]
        #             index_count += 1

        imageIndex = 0
        for j in range(14):
            for k in range(14):
                # data[:, :] = [255, 128, 0]
                # data[j][k] = [dataset[i][imageIndex], dataset[i][imageIndex], dataset[i][imageIndex]]
                data[j][k] = [dataset[i][imageIndex], dataset[i][imageIndex], dataset[i][imageIndex]]
                imageIndex += 1


        # print(data)
        # print(len(data))

        # grey = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        image_dir = "NB15_1_pictures"

        if not os.path.exists("./" + image_dir):
            os.mkdir("./" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("./" + image_dir + "/malicious"):
            os.mkdir("./" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("./" + image_dir + "/normal"):
            os.mkdir("./" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if int(labels[i]) == 1:
            cv2.imwrite("./" + image_dir + "/malicious" + "/record" + str(i) + ".png", data)
        elif int(labels[i]) == 0:
            cv2.imwrite("./" + image_dir + "/normal" + "/record" + str(i) + ".png", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


data = loadDataset()
encoded, label = encodingDataset(data)
makePictures(encoded, label)







