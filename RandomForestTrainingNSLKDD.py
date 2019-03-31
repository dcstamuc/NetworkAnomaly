
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import csv

rf = RandomForestClassifier(n_estimators=5, random_state=10)

train_file_name = "KDDTrain-.csv"
file_name = "KDDTest+.csv"
result_name = "RFTrainMinusTestPlus"

raw_dataset = []
categories = []

num_training = 0
num_testing = 0

with open("../dataset/NSL_KDD-master/" + train_file_name, "rU") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    i = 0
    for row in reader:
        temp = []
        temp.append(row[0])
        for value in row[4:-2]:
            temp.append(value)

        raw_dataset.append(temp)
        del temp

        categories.append(row[1:4])
        num_training += 1

    print(num_training)

with open("../dataset/NSL_KDD-master/" + file_name, "rU") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    i = 0
    for row in reader:
        temp = []
        temp.append(row[0])
        for value in row[4:-2]:
            temp.append(value)

        raw_dataset.append(temp)
        del temp

        categories.append(row[1:4])
        num_testing += 1

    print(num_testing)


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


def getLabels(dataset):

    training_data = dataset[0:num_training]
    testing_data = dataset[num_training:]

    print(len(training_data))
    print(len(testing_data))

    with open("../dataset/NSL_KDD-master/" + train_file_name, "rU") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in reader:

            if row[-2] == 'normal':
                training_data[i].append('normal')
            else:
                training_data[i].append('malicious')

            i += 1

            # if i == 3:
            #     break

    with open("../dataset/NSL_KDD-master/" + file_name, "rU") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in reader:

            if row[-2] == 'normal':
                testing_data[i].append('normal')
            else:
                testing_data[i].append('malicious')

            i += 1

            # if i == 3:
            #     break

    return training_data, testing_data


# def showDataset(dataset):
#     print(dataset[0])
#
#     with open("dataset_KDDTest-.csv", "w") as csvfile:
#         writer = csv.writer(csvfile)
#
#         for i in range(len(dataset)):
#             writer.writerow(dataset[i])


def train_random_forest(training_data, testing_data):
    data = training_data
    testing_data = np.array(testing_data)

    np.random.shuffle(data)

    total = len(data)
    train_len = total * 0.8
    valid_len = total * 0.1
    test_len = total * 0.1

    train_index = int(train_len)
    valid_index = int(train_len + valid_len)
    test_index = int(valid_index + test_len)

    data = np.array(data)

    x_train_input = data[:train_index, 0:-1]
    y_train = data[:train_index, -1]

    x_valid_input = data[train_index:valid_index, 0:-1]
    y_valid = data[train_index:valid_index, -1]

    x_test_input = data[valid_index:, 0:-1]
    y_test = data[valid_index:, -1]

    x_testing_input = testing_data[:, 0:-1]
    y_testing = testing_data[:, -1]

    y_train_input = []
    y_valid_input = []
    y_test_input = []
    y_testing_input = []

    for i in range(len(y_train)):
        if y_train[i] == 'normal':
            y_train_input.append(0)
        else:
            y_train_input.append(1)

    for i in range(len(y_valid)):
        if y_valid[i] == 'normal':
            y_valid_input.append(0)
        else:
            y_valid_input.append(1)

    for i in range(len(y_test)):
        if y_test[i] == 'normal':
            y_test_input.append(0)
        else:
            y_test_input.append(1)

    for i in range(len(y_testing)):
        if y_testing[i] == 'normal':
            y_testing_input.append(0)
        else:
            y_testing_input.append(1)

    y_train_input = np.array(y_train_input)
    y_valid_input = np.array(y_valid_input)
    y_test_input = np.array(y_test_input)

    start_time = datetime.datetime.now()

    rf.fit(x_train_input, y_train_input) # train Random Forest

    end_time = datetime.datetime.now()
    print("Training Duration: " + str(end_time - start_time))

    #################################################################################################

    pred_train = rf.predict(x_train_input)

    train_true = 0
    for i in range(len(pred_train)):
        if pred_train[i] == y_train_input[i]:
            train_true += 1
        # else:
        #     print(pred_train[i])
        #     print(y_train_input[i])

    train_total = len(pred_train)
    train_acc = (train_true * 100 / train_total)

    print("Train accuracy: %.5f%%" % train_acc)

    #################################################################################################

    pred_validation = rf.predict(x_valid_input)
    valid_true = 0
    for i in range(len(pred_validation)):
        if pred_validation[i] == y_valid_input[i]:
            valid_true += 1

    valid_total = len(pred_validation)
    valid_acc = (valid_true * 100 / valid_total)
    print("Validation accuracy: %.5f%%" % valid_acc)

    #################################################################################################

    pred_test = rf.predict(x_test_input)

    test_true = 0
    for i in range(len(pred_test)):
        if pred_test[i] == y_test_input[i]:
            test_true += 1

    test_total = len(pred_test)

    test_acc = (test_true * 100 / test_total)
    print("Test accuracy: %.5f%%" % float(test_acc))

    #################################################################################################
    # above test is that separated from training set as a proportion
    # below testing set is for actual testing set

    pred_testing_set = rf.predict(x_testing_input)
    testing_true = 0

    print(len(pred_testing_set))
    print(len(y_testing_input))
    for i in range(len(pred_testing_set)):
        if pred_testing_set[i] == y_testing_input[i]:
            testing_true += 1

    testing_total = len(pred_testing_set)

    testing_acc = (float(testing_true) * 100) / float(testing_total)
    print("Actual testing accuracy: %.5f%%" % float(testing_acc))

    return y_testing_input, pred_testing_set, testing_acc


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def getStatistics(ms, y_label, y_predicted, acc):
    f1_score_macro = f1_score(y_label, y_predicted, average = "macro")
    f1_score_micro = f1_score(y_label, y_predicted, average = "micro")
    f1_score_weighted = f1_score(y_label, y_predicted, average = "weighted")
    f1_score_none = f1_score(y_label, y_predicted, average = None)

    true_positive = (ms[1][1] / (ms[1][1] + ms[1][0])) * 100
    true_negative = (ms[0][0] / (ms[0][0] + ms[0][1])) * 100
    false_positive = (ms[0][1] / (ms[0][0] + ms[0][1])) * 100
    false_negative = (ms[1][0] / (ms[1][0] + ms[1][1])) * 100

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    # print("F1 score macro: %.2f" % f1_score_macro)
    # print("F1 score micro: %.2f" % f1_score_micro)
    # print("F1 score weighted: %.2f" % f1_score_weighted)
    print("F1 score none:  %.2f" % f1_score_none[1])
    print("True positive: %.2f" % true_positive + "%")
    print("True negative: %.2f" % true_negative + "%")
    print("False positive: %.2f" % false_positive + "%")
    print("False negative: %.2f" % false_negative + "%")

    with open("../results/" + result_name + "_statistical_result.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")

        writer.writerow(["Accuracy: "] + ["%.5f" % acc + "%"])
        writer.writerow(["Precision: "] + ["%.5f" % precision])
        writer.writerow(["Recall: "] + ["%.5f" % recall])
        writer.writerow(["F1 score: "] + ["%.5f" % f1_score_none[1]])
        writer.writerow(["True positive: "] + ["%.5f" % true_positive + "%"])
        writer.writerow(["True negative: "] + ["%.5f" % true_negative + "%"])
        writer.writerow(["False positive: "] + ["%.5f" % false_positive + "%"])
        writer.writerow(["False negative: "] + ["%.5f" % false_negative + "%"])
        writer.writerow(["True positive "] + ["%.d" % ms[1][1]])
        writer.writerow(["True negative "] + ["%.d" % ms[0][0]])
        writer.writerow(["False positive "] + ["%.d" % ms[0][1]])
        writer.writerow(["False negative "] + ["%.d" % ms[1][0]])


def draw_confusion_matrix(y_label, y_predicted, acc):
    cnf_matrix = confusion_matrix(y_label, y_predicted)

    getStatistics(cnf_matrix, y_label, y_predicted, acc)

    np.set_printoptions(precision = 2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = "01", title = result_name + " Confusion Matrix")
    plt.savefig("../results/" + result_name + "_cm.png")
    plt.show()


def draw_roc_curve(label, predicted):

    y_label = np.array(label)
    y_predicted = np.array(predicted)

    fpr, tpr, _ = roc_curve(y_label, y_predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(result_name + "ROC")
    plt.legend(loc="lower right")
    plt.savefig("../results/" + result_name + "_roc.png")
    plt.show()


if __name__ == "__main__":

    len1, categoricalVariables = oneHotForCategorical()
    len2, continuousVariables = oneHotForContinuous()

    dataset = makeMergedDataset(categoricalVariables, continuousVariables)

    trainig_data, testing_data = getLabels(dataset)

    # showDataset(data)

    y_label, y_predicted, acc = train_random_forest(trainig_data, testing_data)

    draw_confusion_matrix(y_label, y_predicted, acc)
    draw_roc_curve(y_label, y_predicted)




