# Importing the libraries
import pandas as pd

# Importing the training dataset
def load_data(train_num, test_num):
    
    train_dataset = pd.read_csv("../IDS2017/IDS2017_"+train_num+".csv")
    test_dataset = pd.read_csv("../IDS2017/IDS2017_"+test_num+".csv")

    # create Xtrain and ytrain
    X_train = pd.DataFrame(train_dataset.iloc[:, : -1].values, columns = train_dataset.columns[:-1])
    y_train = train_dataset.iloc[:, -1].values
    
    # create Xtest and ytest
    X_test = pd.DataFrame(test_dataset.iloc[:, : -1].values, columns = test_dataset.columns[:-1])
    y_test = test_dataset.iloc[:, -1].values
    return X_train, X_test, y_train, y_test  

# feature scalling
def feature_scaling(X_train, X_test, y_train, y_test):
    from sklearn import preprocessing
    minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X_train = pd.DataFrame(minmaxscaler.fit_transform(X_train), columns = X_train.columns[:])
    X_test = pd.DataFrame(minmaxscaler.fit_transform(X_test), columns = X_test.columns[:])
    return X_train, X_test

# implementation
def svm(file_num,classifier,clf):
    import time
    from sklearn.metrics import accuracy_score, confusion_matrix
    train_start = time.time()
    classifier.fit(X_train, y_train)
    train_end = time.time()
    train_time = train_end - train_start
    print(train_time)

    test_start = time.time()
    y_pred = classifier.predict(X_test)
    test_end = time.time()
    test_time = test_end - test_start
    print(test_time)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*((precision*recall)/(precision+recall))
    print('precision = ', precision)
    print('recall = ', recall)
    print('F1-score = ', f1_score)
    
    
    n=78
    df = pd.DataFrame({'classifier':clf, 'number of features':n, 'train_time':train_time, 'test_time':test_time,
                       'accuracy':accuracy,'tn':tn,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'F-1score':f1_score}, index=[0])
    return df           

from sklearn.svm import SVC
number = ["01","02","03","04","05","06","07","08","09","10"]
for i in range(0,len(number),2):
    train_num=number[i]
    test_num=number[i+1]
    X_train, X_test, y_train, y_test = load_data(train_num,test_num)
    file_num=number[i]
    
    print(number[i])
    print(number[i+1])
    
    clf="SVM"
    classifier = SVC(kernel='linear', random_state=0)
    tf_svm = svm(file_num,classifier,clf)


#    tf_svm.to_csv('ids_'+file_num+'.csv')