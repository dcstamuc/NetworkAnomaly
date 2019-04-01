
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

train_file_name = "final_NB15_training-set.csv"
test_file_name = "final_NB15_testing-set.csv"
result_name = "RF_NB15"

num_training = 0
num_testing = 0


def loadDataset():
    training_set = []
    testing_set = []

    with open("../dataset/UNSW_RandomForest/" + train_file_name, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            training_set.append(row)

    num_training = len(training_set)

    with open("../dataset/UNSW_RandomForest/" + test_file_name, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            testing_set.append(row)

    num_testing = len(testing_set)

    return training_set, testing_set


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
    y_train = data[:train_index, -1].astype('int')

    x_valid_input = data[train_index:valid_index, 0:-1]
    y_valid = data[train_index:valid_index, -1].astype('int')

    x_test_input = data[valid_index:, 0:-1]
    y_test = data[valid_index:, -1].astype('int')

    x_testing_input = testing_data[:, 0:-1]
    y_testing = testing_data[:, -1].astype('int')

    start_time = datetime.datetime.now()

    rf.fit(x_train_input, y_train) # train Random Forest

    end_time = datetime.datetime.now()
    print("Training Duration: " + str(end_time - start_time))

    #################################################################################################

    pred_train = rf.predict(x_train_input)

    train_true = 0
    for i in range(len(pred_train)):
        if pred_train[i] == y_train[i]:
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
        if pred_validation[i] == y_valid[i]:
            valid_true += 1

    valid_total = len(pred_validation)
    valid_acc = (valid_true * 100 / valid_total)
    print("Validation accuracy: %.5f%%" % valid_acc)

    #################################################################################################

    pred_test = rf.predict(x_test_input)

    test_true = 0
    for i in range(len(pred_test)):
        if pred_test[i] == y_test[i]:
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
    print(len(y_testing))
    for i in range(len(pred_testing_set)):
        if pred_testing_set[i] == y_testing[i]:
            testing_true += 1

    testing_total = len(pred_testing_set)

    testing_acc = (float(testing_true) * 100) / float(testing_total)
    print("Actual testing accuracy: %.5f%%" % float(testing_acc))

    print(type(y_testing[0]))
    print(type(pred_testing_set[0]))

    return y_testing, pred_testing_set, testing_acc


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

    training_data, testing_data = loadDataset()

    y_label, y_predicted, acc = train_random_forest(training_data, testing_data)

    draw_confusion_matrix(y_label, y_predicted, acc)
    draw_roc_curve(y_label, y_predicted)




