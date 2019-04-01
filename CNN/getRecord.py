
import csv

# with open("./UNSW_NB15_training-set_full_x_attack_cat.csv", "rU") as csv_reader:
#     reader = csv.reader(csv_reader, delimiter=',')
#
#     i = 0
#     for row in reader:
#         i += 1
#
#     print(i)

with open("../dataset/UNSW/UNSW_NB15_training-set.csv", "rU") as csv_reader:
    reader = csv.reader(csv_reader, delimiter=',')

    i = 0
    for row in reader:
        i += 1

    print(i)
