
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import f1_score, roc_curve, auc

rf = RandomForestClassifier(n_estimators=5, random_state=10)

dataset_name = "final_mawilab_dataset_"
result_name = "RF_EACH_MAWILAB_"


def load_dataset():
    print("============================ loading dataset ============================")

    dataset = []

    for i in range(4):
        with open("../dataset/MAWILAB/" + dataset_name + str(i) + ".csv", "r") as csv_reader:
            reader = csv.reader(csv_reader, delimiter=',')

            temp = []
            for row in reader:
                temp.append(row)

            dataset.append(temp)

    print("Number of dataset: " + str(len(dataset)))

    for i in range(len(dataset)):
        print("Number of each dataset's record: " + str(len(dataset[i])))

    return dataset


def train(dataset):
    print("============================ training random forest ============================")

    train_len = len(dataset[0]) * 0.8
    valid_len = len(dataset[0]) * 0.1
    test_len = len(dataset[0]) * 0.1

    train_index = int(train_len)
    valid_index = int(train_len + valid_len)
    test_index = int(valid_index + test_len)

    print("training set training end index: " + str(train_index))
    print("training set validation end index: " + str(valid_index))
    print("training set testing end index: " + str(test_index))

    train_data = np.array(dataset[0])

    training = train_data[:train_index, 0:-1]
    training_temp_label = train_data[:train_index, -1]

    validation = train_data[train_index:valid_index, 0:-1]
    validation_temp_label = train_data[train_index:valid_index, -1]

    train_testing = train_data[valid_index:, 0:-1]
    train_testing_temp_label = train_data[valid_index:, -1]

    print(len(training))
    print(len(validation))
    print(len(train_testing))

    training_label = []
    validation_label = []
    train_testing_label = []

    for i in range(len(training_temp_label)):
        if training_temp_label[i] == "normal":
            training_label.append(0)
        elif training_temp_label[i] == "anomaly":
            training_label.append(1)

    for i in range(len(validation_temp_label)):
        if validation_temp_label[i] == "normal":
            validation_label.append(0)
        elif validation_temp_label[i] == "anomaly":
            validation_label.append(1)

    for i in range(len(train_testing_temp_label)):
        if train_testing_temp_label[i] == "normal":
            train_testing_label.append(0)
        elif train_testing_temp_label[i] == "anomaly":
            train_testing_label.append(1)

    print("Number of train label: " + str(len(training_label)))
    print("Number of validation label: " + str(len(validation_label)))
    print("Number of train test label: " + str(len(train_testing_label)))

    start_time = datetime.datetime.now()

    rf.fit(training, training_label)

    end_time = datetime.datetime.now()
    print("Training Duration: " + str(end_time - start_time))

    for step in range(len(dataset[1:])):
        data = np.array(dataset[step + 1])
        testing = data[:, 0:-1]
        testing_temp_label = data[:, -1]

        print("Number of testing records: " + str(len(testing)))
        print("Number of testing label: " + str(len(testing_temp_label)))

        testing_label = []

        for i in range(len(testing_temp_label)):
            if testing_temp_label[i] == "normal":
                testing_label.append(0)
            else:
                testing_label.append(1)

        predicted = rf.predict(testing)

        test_true = 0
        test_total = len(predicted)

        for i in range(len(predicted)):
            if predicted[i] == testing_label[i]:
                test_true += 1

        acc = (float(test_true) * 100) / float(test_total)
        print("Testing accuracy: %.5f%%" % float(acc))

        print("Current step: " + str(step + 1))

        draw_confusion_matrix(testing_label, predicted, acc, step + 1)
        draw_roc_curve(testing_label, predicted, step + 1)


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


def getStatistics(ms, y_label, y_predicted, acc, step):
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

    with open("../results/" + result_name + "_" + str(step) + "_statistical_result.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

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


def draw_confusion_matrix(y_label, y_predicted, acc, step):
    cnf_matrix = confusion_matrix(y_label, y_predicted)

    getStatistics(cnf_matrix, y_label, y_predicted, acc, step)

    np.set_printoptions(precision = 2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = "01", title = result_name + " Confusion Matrix")
    plt.savefig("../results/" + result_name + "_" + str(step) + "_cm.png")
    plt.show()


def draw_roc_curve(label, predicted, step):

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
    plt.savefig("../results/" + result_name + "_" + str(step) + "_roc.png")
    plt.show()


if __name__ == "__main__":
    dataset = load_dataset()
    train(dataset)









