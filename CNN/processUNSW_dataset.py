
import csv
import numpy as np

dataset = []

with open("./UNSW/UNSW-NB15_4.csv", "rU") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    x = 0
    for row in reader:
        temp = []

        for i in range(len(row[4:9])):
            temp.append(row[i + 4])

        for i in range(len(row[13:18])):
            temp.append(row[i + 13])

        for i in range(len(row[30:35])):
            temp.append(row[i + 30])

        temp.append(row[48])
        # print(temp)
        dataset.append(temp)

with open("./ProcessedDataset/pre_NB15_4.csv", "w") as csvfile:
    writer = csv.writer(csvfile)

    for i in range(len(dataset)):
        writer.writerow(dataset[i])



