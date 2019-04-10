
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2
import sys
import os

dataset_name = sys.argv[1]
raw_dataset = []
categories = []

with open("../dataset/NSL_KDD-master/" + dataset_name + ".csv", "rU") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    label = []

    i = 0
    for row in reader:
        temp = []
        temp.append(row[0])
        for value in row[4:-2]:
            temp.append(value)

        raw_dataset.append(temp)
        del temp

        categories.append(row[1:4])

        label.append(row[-2])
        i += 1
        if i == 10:
            break

print(raw_dataset[0])
print(categories[0])
print(len(raw_dataset[0]))
print(len(categories[0]))
print("-----------------------------------")
print(len(raw_dataset))
print(len(label))
print("-----------------------------------")
print(len(raw_dataset))
print(len(categories))

##
# The first thing is to convert continuous values to one hot encoded
# The second thing is to convert categorical values to one hot encoded
# The third thing is to merge them together
# index 1, 2, and 3 should be one hot encoded categorical values
# #


def oneHotForCategorical():
    global categories

    dataset = []

    column1 = {}
    column2 = {}
    column3 = {}

    for row in categories:
        dataset.append(row)
        if column1.get(row[0]) == None:
            column1[row[0]] = 1
        else:
            column1[row[0]] += 1

        if column2.get(row[1]) == None:
            column2[row[1]] = 1
        else:
            column2[row[1]] += 1

        if column3.get(row[2]) == None:
            column3[row[2]] = 1
        else:
            column3[row[2]] += 1

    # print(column1)
    # print(column2)
    # print(column3)

    new_dict = {}
    sorted_dict = {}

    temp = list(column2.keys())
    temp2 = list(column2.values())

    # for i in range(70):
    #     new_dict[temp2[i]] = temp[i]

    # print("=========================================================")
    # print(new_dict)

    # keys_values = list(new_dict.keys())
    # print(keys_values)
    #
    # sorted_keys = sorted(keys_values)

    dict_sorted = sorted(new_dict, reverse=True)
    # print(dict_sorted)

    final_column3 = {}

    # print("")

    for i in range(len(dict_sorted)):
        final_column3[dict_sorted[i]] = new_dict[dict_sorted[i]]

    # print(final_column3)

    oh_2 = []
    oh_3 = []
    oh_4 = []

    encoded_dataset = []
    # print("===================================================================")
    # print(dataset[0])
    # print(dataset[0][2:-1])
    # print("===================================================================")

    for i in range(len(dataset)): # for column 2
        encoded_dataset.append(dataset[i][0:1])
        # print(encoded_dataset[i])

        if dataset[i][0] == 'tcp':
            encoded_dataset[i][0] = 1
            for j in [0, 0]:
                encoded_dataset[i].append(j)

            # for k in range(len(dataset[i][2:-1])):
            #     encoded_dataset[i].append(dataset[i][2 + k])

        elif dataset[i][0] == 'udp':
            encoded_dataset[i][0] = 0
            for j in [1, 0]:
                encoded_dataset[i].append(j)

            # for k in range(len(dataset[i][2:-1])):
            #     encoded_dataset[i].append(dataset[i][2 + k])


        else:
            encoded_dataset[i][0] = 0
            for j in [0, 1]:
                encoded_dataset[i].append(j)

            # for k in range(len(dataset[i][2:-1])):
            #     encoded_dataset[i].append(dataset[i][2 + k])


    # print(encoded_dataset[0])
    # print(encoded_dataset[1])
    # print(encoded_dataset[2])
    # print(encoded_dataset[3])
    # print(encoded_dataset[4])
    # print("===================================================================")

    # print(dataset[0])
    for i in range(len(dataset)): # for column 3
        if dataset[i][1] == 'http':
            for j in range(70):
                if j == 0:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][1] == 'private':
            for j in range(70):
                if j == 1:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][1] == 'domain_u':
            for j in range(70):
                if j == 2:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][1] == 'smtp':
            for j in range(70):
                if j == 3:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][1] == 'ftp_data':
            for j in range(70):
                if j == 4:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][1] == 'eco_i':
            for j in range(70):
                if j == 5:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'other':
            for j in range(70):
                if j == 6:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'remote_job':
            for j in range(70):
                if j == 7:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'name':
            for j in range(70):
                if j == 8:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'netbios_ns':
            for j in range(70):
                if j == 9:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'mtp':
            for j in range(70):
                if j == 10:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'telnet':
            for j in range(70):
                if j == 11:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'finger':
            for j in range(70):
                if j == 12:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'supdup':
            for j in range(70):
                if j == 13:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'uucp_path':
            for j in range(70):
                if j == 14:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'Z39_50':
            for j in range(70):
                if j == 15:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'csnet_ns':
            for j in range(70):
                if j == 16:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'uucp':
            for j in range(70):
                if j == 17:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'netbios_dgm':
            for j in range(70):
                if j == 18:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'urp_i':
            for j in range(70):
                if j == 19:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'auth':
            for j in range(70):
                if j == 20:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'domain':
            for j in range(70):
                if j == 21:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'ftp':
            for j in range(70):
                if j == 22:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'bgp':
            for j in range(70):
                if j == 23:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'ldap':
            for j in range(70):
                if j == 24:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'ecr_i':
            for j in range(70):
                if j == 25:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'gopher':
            for j in range(70):
                if j == 26:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'vmnet':
            for j in range(70):
                if j == 27:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'systat':
            for j in range(70):
                if j == 28:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'http_443':
            for j in range(70):
                if j == 29:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'efs':
            for j in range(70):
                if j == 30:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'whois':
            for j in range(70):
                if j == 31:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'imap4':
            for j in range(70):
                if j == 32:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'iso_tsap':
            for j in range(70):
                if j == 33:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'echo':
            for j in range(70):
                if j == 34:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'klogin':
            for j in range(70):
                if j == 35:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'link':
            for j in range(70):
                if j == 36:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'sunrpc':
            for j in range(70):
                if j == 37:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'login':
            for j in range(70):
                if j == 38:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'kshell':
            for j in range(70):
                if j == 39:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'sql_net':
            for j in range(70):
                if j == 40:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'time':
            for j in range(70):
                if j == 41:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'hostnames':
            for j in range(70):
                if j == 42:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'exec':
            for j in range(70):
                if j == 43:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'ntp_u':
            for j in range(70):
                if j == 44:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'discard':
            for j in range(70):
                if j == 45:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'nntp':
            for j in range(70):
                if j == 46:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'courier':
            for j in range(70):
                if j == 47:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'ctf':
            for j in range(70):
                if j == 48:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'ssh':
            for j in range(70):
                if j == 49:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'daytime':
            for j in range(70):
                if j == 50:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'shell':
            for j in range(70):
                if j == 51:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'netstat':
            for j in range(70):
                if j == 52:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'pop_3':
            for j in range(70):
                if j == 53:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'nnsp':
            for j in range(70):
                if j == 54:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'IRC':
            for j in range(70):
                if j == 55:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'pop_2':
            for j in range(70):
                if j == 56:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'printer':
            for j in range(70):
                if j == 57:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'tim_i':
            for j in range(70):
                if j == 58:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'pm_dump':
            for j in range(70):
                if j == 59:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'red_i':
            for j in range(70):
                if j == 60:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'netbios_ssn':
            for j in range(70):
                if j == 61:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'rje':
            for j in range(70):
                if j == 62:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'X11':
            for j in range(70):
                if j == 63:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'urh_i':
            for j in range(70):
                if j == 64:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'http_8001':
            for j in range(70):
                if j == 65:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'aol':
            for j in range(70):
                if j == 66:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'http_2784':
            for j in range(70):
                if j == 67:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'tftp_u':
            for j in range(70):
                if j == 68:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][1] == 'harvest':
            for j in range(70):
                if j == 69:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)


    # print(encoded_dataset[0])
    # print(encoded_dataset[1])
    # print(encoded_dataset[2])
    # print(encoded_dataset[3])
    # print(encoded_dataset[4])

    print("===================================================================")

    # print(dataset[0])
    for i in range(len(dataset)):
        if dataset[i][2] == 'SF':
            for j in [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'S0':
            for j in [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'REJ':
            for j in [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'RSTR':
            for j in [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'SH':
            for j in [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'RSTO':
            for j in [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'S1':
            for j in [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'RSTOS0':
            for j in [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'S3':
            for j in [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'S2':
            for j in [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'OTH':
            for j in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
                encoded_dataset[i].append(j)


    # print("---------------------------------------------------------------------------")
    # print(dataset[0])
    # print(encoded_dataset[0])
    # print(len(encoded_dataset[0]))
    # print(len(encoded_dataset))
    print("Categorical encoded", encoded_dataset[0])
    print("Categorical encoded length", len(encoded_dataset[0]))

    return len(encoded_dataset), encoded_dataset


def oneHotForContinuous():
    global raw_dataset
    raw_dataset = np.array(raw_dataset)

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(raw_dataset)
    print("Normalized", normalized[0])
    print("Normalized length", len(normalized[0]))

    # print(normalised[0])
    # print(len(normalised[0]))

    return len(normalized), normalized


def makeMergedDataset(categoricalVariables, continuousVariables):

    dataset = []

    for i in range(len(categoricalVariables)):
        temp = []

        temp.append(continuousVariables[i][0])

        for j in range(len(categoricalVariables[i])):
            temp.append(categoricalVariables[i][j])

        for k in continuousVariables[i][1:]:
            temp.append(k)

        dataset.append(temp)
        del temp

    # print(dataset[0])
    # print(dataset[10])
    # print(dataset[100])
    # print(dataset[1000])
    # print(dataset[10000])
    print(len(dataset[0]))
    print(dataset[0])
    # print(len(dataset))

    return dataset


def addPadding(dataset):
    print(len(dataset[0]))
    print(np.array(dataset[0]))

    for i in range(len(dataset)):
        for j in range(48):
            dataset[i].append(0)

    # print(dataset[0])
    # print(len(dataset[0]))
    # print(dataset[0:2])
    # print(dataset)

    print(len(dataset[0]))
    print(len(dataset))

    with open("NSLKDD_Image_Array.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(len(dataset)):
            writer.writerow(dataset[i])

    return dataset


def newEncodingScheme(dataset):
    print("==================== encoding ====================")
    dataset = dataset
    bit_size = 24

    print(dataset[0])
    print(len(dataset[0]))

    encoded_dataset = []

    for i in range(len(dataset)):
        temp = []

        # if i == 1:
        #     break

        for j in range(len(dataset[i])):
            if float(dataset[i][j]) == 1.0:
                temp.append(pow(2, bit_size) - 1)
            elif float(dataset[i][j]) == 0.0:
                temp.append(0)
            else:
                temp.append(int(dataset[i][j] * pow(2, bit_size) - 1))

        for k in range(22):
            temp.append(0)

        encoded_dataset.append(temp)

    print("===================================================")
    print(encoded_dataset[0])
    print(len(encoded_dataset[0]))

    return encoded_dataset


def makeImages(dataset):
    print("======================================= generate images ========================================")
    for i in range(len(dataset)):
        # if i == 1:
        #     break

        # print(dataset[i])

        data = np.zeros([12, 12, 3], dtype=np.uint8)

        imageIndex = 0
        for j in range(12):
            for k in range(12):
                # encoded with 24 bits
                temp_encoded_number = '{0:024b}'.format(dataset[i][imageIndex])

                # each 8 bits stored in each color variable
                R = int(temp_encoded_number[0:8], 2)
                G = int(temp_encoded_number[8:16], 2)
                B = int(temp_encoded_number[16:24], 2)

                data[j][k] = [R, G, B]

                imageIndex += 1


        # print(data)
        # print(len(data))

        # grey = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        image_dir = "NSL-KDD_" + dataset_name

        if not os.path.exists("../generatedImages/NSL-KDD_Our"):
            os.mkdir("../generatedImages/NSL-KDD_Our")

        if not os.path.exists("../generatedImages/NSL-KDD_Our/" + image_dir):
            os.mkdir("../generatedImages/NSL-KDD_Our/" + image_dir)
            print(image_dir + " directory is created")

        if not os.path.exists("../generatedImages/NSL-KDD_Our/" + image_dir + "/malicious"):
            os.mkdir("../generatedImages/NSL-KDD_Our/" + image_dir + "/malicious")
            print(image_dir + "/malicious " + " directory is created")

        if not os.path.exists("../generatedImages/NSL-KDD_Our/" + image_dir + "/normal"):
            os.mkdir("../generatedImages/NSL-KDD_Our/" + image_dir + "/normal")
            print(image_dir + "/normal " + " directory is created")

        if label[i] != "normal":
            cv2.imwrite("../generatedImages/NSL-KDD_Our/" + image_dir + "/malicious" + "/record" + str(i) + ".png", data)
        elif label[i] == "normal":
            cv2.imwrite("../generatedImages/NSL-KDD_Our/" + image_dir + "/normal" + "/record" + str(i) + ".png", data)

        if i % 10000 == 0 and i != 0:
            print(str(i) + "th image created")


len1, categoricalVariables = oneHotForCategorical()
len2, continuousVariables = oneHotForContinuous()

dataset = makeMergedDataset(categoricalVariables, continuousVariables)
encoded_dataset = newEncodingScheme(dataset)
makeImages(encoded_dataset)















