
import csv

file_name = "final_IDS2017_M2F_1M_Comma_"


def load_dataset():

    training_set = []
    testing_set = []

    for i in range(10):
        with open("../dataset/IDS/" + file_name + str(i) + ".csv", "r") as csvReader:
            if i % 2 == 0:
                reader = csv.reader(csvReader, delimiter=',')

                for row in reader:
                    training_set.append(row)

            else:
                reader = csv.reader(csvReader, delimiter=',')

                for row in reader:
                    testing_set.append(row)

    print("Loading dataset is done")
    print("Number of training set: " + str(len(training_set)))
    print("Number of testing set: " + str(len(testing_set)))

    return training_set, testing_set


def generate_dataset(training_set, testing_set):
    with open("../dataset/IDS/IDS_training_set.csv", "w") as csvWriter:
        writer = csv.writer(csvWriter)

        for i in range(len(training_set)):
            writer.writerow(training_set[i])

    with open("../dataset/IDS/IDS_testing_set.csv", "w") as csvWriter:
        writer = csv.writer(csvWriter)

        for i in range(len(testing_set)):
            writer.writerow(testing_set[i])


if __name__ == "__main__":
    train, test = load_dataset()
    generate_dataset(train, test)






