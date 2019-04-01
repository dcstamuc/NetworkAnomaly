
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import itertools

dirName = "UNSW_NB15_results"
file_name = "24_UNSW"

def loadResultSet():

    malicious_total_cnt = 0
    malicious_total_list = []
    malicious_list = []
    with open("../dataset/" + dirName + "/" + file_name + "_malicious_result.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            malicious_list.append(round(float(row[2])))
            malicious_total_cnt += 1
            malicious_total_list.append(1)

    normal_total_cnt = 0
    normal_total_list = []
    normal_list = []
    with open("../dataset/" + dirName + "/" + file_name + "_normal_result.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            normal_list.append(round(float(row[3])))
            normal_total_cnt += 1
            normal_total_list.append(0)

    return malicious_total_cnt, malicious_total_list, malicious_list, normal_total_cnt, normal_total_list, normal_list


def showInformation(m_total_cnt, m_total_list, m_list, n_total_cnt, n_total_list, n_list):
    malicious_cnt = 0
    normal_cnt = 0

    for i in range(len(m_list)):
        if m_list[i] == 1:
            malicious_cnt += 1

    for i in range(len(n_list)):
        if n_list[i] == 1:
            normal_cnt += 1

    print("Total malicious count: %d" % m_total_cnt)
    print("Predicted malicious count: %d" % malicious_cnt)
    print("Total normal count: %d" % n_total_cnt)
    print("Predicted normal count: %d" % normal_cnt)


def putLabelsTogether(m_total_list, n_total_list, m_list, n_list):

    for i in range(len(n_total_list)):
        m_total_list.append(n_total_list[i])

    for i in range(len(n_list)):
        if n_list[i] == 1:
            m_list.append(0)
        elif n_list[i] == 0:
            m_list.append(1)

    y_label = m_total_list
    y_predicted = m_list

    print(len(y_label))
    print(len(y_predicted))

    m_cnt = 0
    n_cnt = 0

    sum = 0
    for i in range(len(y_label)):
        if y_label[i] == y_predicted[i]:
            sum += 1

    for i in range(len(y_label)):
        if y_label[i] == 1 and y_predicted[i] == 1:
            m_cnt += 1

    for i in range(len(y_label)):
        if y_label[i] == 0 and y_predicted[i] == 0:
            n_cnt += 1

    accuracy = float(sum) * 100 / float(len(y_label))
    print("Accuracy: %.5f%%" % accuracy)
    print("Number of malicious: " + str(m_cnt))
    print("Number of normal: " + str(n_cnt))

    return y_label, y_predicted, accuracy



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
    print("F1 score none: %.2f" % f1_score_none[1])
    print("True positive: %.2f" % true_positive + "%")
    print("True negative: %.2f" % true_negative + "%")
    print("False positive: %.2f" % false_positive + "%")
    print("False negative: %.2f" % false_negative + "%")

    with open("../results/" + file_name + "_statistical_result.csv", "w") as csvfile:
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
    plot_confusion_matrix(cnf_matrix, classes = "01", title = file_name + " Confusion Matrix")
    plt.savefig("../results/" + file_name + "_cm.png")
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
    plt.title(file_name + ' ROC')
    plt.legend(loc="lower right")
    plt.savefig("../results/" + file_name + "_roc.png")
    plt.show()


malicious_total_cnt, malicious_total_list, malicious_list, normal_total_cnt, normal_total_list, normal_list = loadResultSet()
showInformation(malicious_total_cnt, malicious_total_list, malicious_list,
                normal_total_cnt, normal_total_list, normal_list)
y_label, y_predicted, acc = putLabelsTogether(malicious_total_list, normal_total_list, malicious_list, normal_list)

draw_confusion_matrix(y_label, y_predicted, acc)
draw_roc_curve(y_label, y_predicted)


