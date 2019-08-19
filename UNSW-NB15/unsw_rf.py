# Importing the libraries
import pandas as pd



# Importing the training dataset
def load_data():
    
    train_dataset = pd.read_csv("..unsw_nb15/UNSW_NB15_training-set.csv")
    test_dataset = pd.read_csv("../unsw_nb15/UNSW_NB15_testing-set.csv")

    train_dataset.drop(["id","proto","service","state","attack_cat"], axis = 1, inplace = True)
    test_dataset.drop(["id","proto","service","state","attack_cat"], axis = 1, inplace = True)
    
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
def rf(classifier,clf):
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
    
    
    n=39
    df = pd.DataFrame({'classifier':clf, 'number of features':n, 'train_time':train_time, 'test_time':test_time,
                       'accuracy':accuracy,'tn':tn,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'F-1score':f1_score}, index=[0])

    return df           

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = feature_scaling(X_train, X_test, y_train, y_test)

clf="RF"
classifier = RandomForestClassifier(n_estimators = 5, random_state=0)
tf_rf = rf(classifier,clf)

#tf_rf.to_csv('unsw_svc.csv')
