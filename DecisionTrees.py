# DT for ASM2

from sklearn.model_selection import KFold
import time
import scipy as sc
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectPercentile, f_classif
from sklearn.feature_selection import chi2
from sklearn import preprocessing, tree
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib.legend_handler import HandlerLine2D
import matplotlib as mpl
import matplotlib.pyplot as plt

global Tag_names_train, Tag_names_valid, Tag_names_test
global Tt_names_train, Tt_names_valid, Tt_names_test
global convert_tb


# Importing Dataset
def importData():
    X_train = pd.read_csv('./Data/train_features.tsv', sep='\t')
    X_valid = pd.read_csv('./Data/valid_features.tsv', sep='\t')
    X_test = pd.read_csv('./Data/NEW_test_features.tsv', sep='\t')
    y_train = pd.read_csv('./Data/train_labels.tsv', sep='\t')
    y_valid = pd.read_csv('./Data/valid_labels.tsv', sep='\t')

    # ConvertNumeric
    y_train = ConvertNumericLabel(y_train)  # df 2// movieId// genres
    y_valid = ConvertNumericLabel(y_valid)

    X_train, X_orig_train, X_Tag_df, X_Title_df = ConvertNumeric(X_train, "train")
    X_valid, X_orig_valid, X_Tag_df_valid, X_Title_df_valid = ConvertNumeric(X_valid, "valid")
    X_test, X_orig_test, X_Tag_df_test, X_Title_df_test = ConvertNumeric(X_test, "test")

    # ProcessFeatureLength
    X_valid = ProcessFeatureLength(X_valid, "valid")
    X_test = ProcessFeatureLength(X_test, "test")

    return X_train, X_valid, X_test, y_train, y_valid


def Ft_Sel(X_train, X_valid, X_test, y_train, y_valid, k=600):
    # FeatureSelection
    X_train = preprocessing.MinMaxScaler().fit_transform(X_train)  # normalize data to [0,1]  ##Chi2 cannot get negative value
    X_train, sel_cols = FeatureSelection_1(X_train, y_train,k)

    # Drop all cols except sel_cols
    X_valid = preprocessing.MinMaxScaler().fit_transform(X_valid)
    X_valid = FeatureSelection_Test(X_valid, sel_cols)
    X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
    X_test = FeatureSelection_Test(X_test, sel_cols)
    # arr to df
    X_train = pd.DataFrame(data=X_train[0:, 0:], index=[i for i in range(X_train.shape[0])],
                           columns=[i for i in range(X_train.shape[1])])
    X_valid = pd.DataFrame(data=X_valid[0:, 0:], index=[i for i in range(X_valid.shape[0])],
                           columns=[i for i in range(X_valid.shape[1])])
    X_test = pd.DataFrame(data=X_test[0:, 0:], index=[i for i in range(X_test.shape[0])],
                          columns=[i for i in range(X_test.shape[1])])

    return X_train, X_valid, X_test, y_train, y_valid


# X: Convert string data to numeric values
def ConvertNumeric(df, case):
    print("ConvertNumeric: Convert string data to numeric values")
    # Year Column
    # Tag Column
    df_tag = df.iloc[:, 4]
    df_tag = df_tag.str.replace(',', ' ')
    # Title Column
    df_title = df.iloc[:, 1]
    df_title = df_title.str.replace(',', '');
    df_title = df_title.str.replace(')', '');
    df_title = df_title.str.replace('(', '');
    df_title = df_title.str.replace(':', '');
    df_title = df_title.str.replace('&', '');

    ###### Process Tag#######################################
    # CountVector
    vectorizer = CountVectorizer()
    A = vectorizer.fit_transform(df_tag)  # csr-matrix
    # vectorizer.vocabulary_.get(u'algorithm')

    # tf–idf
    tfidf_transformer = TfidfTransformer()
    A = tfidf_transformer.fit_transform(A)

    #print("Tags", len(vectorizer.get_feature_names()), "--", vectorizer.get_feature_names())

    # for later use in ProcessFeature()
    global Tag_names_train, Tag_names_valid, Tag_names_test
    if (case == "train"):
        Tag_names_train = vectorizer.get_feature_names()
    elif (case == "valid"):
        Tag_names_valid = vectorizer.get_feature_names()
    elif (case == "test"):
        Tag_names_test = vectorizer.get_feature_names()

    Tag_df = pd.DataFrame(data=A.toarray()[0:, 0:], index=[i for i in range(A.toarray().shape[0])], columns=['Tg' + str(i) for i in range(A.toarray().shape[1])])

    ######Process Title#######################################

    # (0) handle 3 NaN element
    # sum = df_title.isnull().sum()
    # print("NaN value(Title): ", sum) #handle 3 NaN element

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=" ")  # strategy = 'most_frequent'
    imp = imp.fit(df_title.to_numpy().reshape(-1, 1))
    df_title = imp.transform(df_title.to_numpy().reshape(-1, 1)).ravel()  # Nan to constant
    df_title = df_title.reshape(-1, 1)
    # sum = df_title.isnull().sum()
    # print(sum) #Nan element->0 after processing

    ## Text to Number
        # arry to df
    df_title = pd.DataFrame(data=df_title[0:, 0:], index=[i for i in range(df_title.shape[0])], columns=['Tt' + str(i) for i in range(df_title.shape[1])])
        # df to ser
    df_title = df_title.iloc[:, 0]
    # Count vector
    B = vectorizer.fit_transform(df_title)
    # tf–idf
    B = tfidf_transformer.fit_transform(B)

    #print("Titles", len(vectorizer.get_feature_names()), "--", vectorizer.get_feature_names())
    Title_df = pd.DataFrame(data=B.toarray()[0:, 0:], index=[i for i in range(B.toarray().shape[0])],
                            columns=['Tt' + str(i) for i in range(B.toarray().shape[1])])

    # for later use in ProcessFeature()
    global Tt_names_train, Tt_names_valid, Tt_names_test
    if (case == "train"):
        Tt_names_train = vectorizer.get_feature_names()
    elif (case == "valid"):
        Tt_names_valid = vectorizer.get_feature_names()
    elif (case == "test"):
        Tt_names_test = vectorizer.get_feature_names()

    df_orig = df.drop(columns=['title', 'YTId', 'tag'])

    ## Year : str >> float
    err_idx = []
    for x in range(df_orig.iloc[:, 1].size):
        try:
            df_orig.iloc[:, 1].at[x] = float(df_orig.iloc[:, 1].at[x])
        except ValueError:
            err_idx.append(x)
            df_orig.iloc[:, 1].at[x] = float(0)
        else:
            df_orig.iloc[:, 1].at[x] = float(df_orig.iloc[:, 1].at[x])
    # missing year val << avg value assign
    avg = df_orig.iloc[:, 1].mean()
    for idx in err_idx:
        df_orig.iloc[:, 1].at[idx] = avg

    # [4] Concat Numerized columns with original table
    df_numeric = pd.concat([df_orig, Tag_df, Title_df], axis=1, sort=False)

    return df_numeric, df_orig, Tag_df, Title_df


def FeatureSelection_1(X_train, y_train, k=600):
    print("FeatureSelection_1")
    print(X_train.shape)
    kb = SelectKBest(chi2, k=k)
    #kb = SelectKBest(mutual_info_classif, k=k)
    #kb = SelectKBest(f_classif, k=k)

    X_train = kb.fit_transform(X_train, y_train.iloc[:, 1])

    selected_ft = kb.get_support()  # list of booleans of selected res
    #print(selected_ft)

    sel_cols = []
    i = 0
    for bool in selected_ft:
        if bool:
            sel_cols.append(i)
        i += 1

    #print(sel_cols)  # outof 6353
    #print(X_train.shape)

    return X_train, sel_cols


def FeatureSelection_Test(Xset, sel_cols):  # ndarr

    new_arr = np.zeros((Xset.shape[0], sel_cols.__len__()))

    i = 0
    for col in sel_cols:
        new_arr[:, i] = Xset[:, col]
        i += 1

    return new_arr


# Process feature - tag/title feature 개수 맞추기 - 여기서 시간이 엄청 걸리는거 같은데??
def ProcessFeatureLength(df, case):
    global Tag_names_train, Tag_names_valid, Tag_names_test
    global Tt_names_train, Tt_names_valid, Tt_names_test
    size_Tg = len(Tag_names_train)  # list
    size_Tt = len(Tt_names_train)

    cols = []

    # train data set feature size
    for num in range(size_Tg):
        cols.append("Tg" + str(num))
    for num in range(size_Tt):
        cols.append("Tt" + str(num))
    new_df = pd.DataFrame(columns=cols)

    # Process valid data set to train set size
    if (case == "valid"):
        # Tags
        for y in range(len(Tag_names_valid)):
            for x in range(len(Tag_names_train)):
                if Tag_names_valid[y] == Tag_names_train[x]:
                    new_df['Tg' + str(x)] = df.iloc[:, 129 + y]
        # Titles
        for y in range(len(Tt_names_valid)):
            for x in range(len(Tt_names_train)):
                if Tt_names_valid[y] == Tt_names_train[x]:
                    new_df['Tt' + str(x)] = df.iloc[:, 129 + len(Tag_names_valid) + y]

    # Process test data set to train set size
    if (case == "test"):
        # Tags
        for y in range(len(Tag_names_test)):
            for x in range(len(Tag_names_train)):
                if Tag_names_test[y] == Tag_names_train[x]:
                    new_df['Tg' + str(x)] = df.iloc[:, 129 + y]
        # Titles
        for y in range(len(Tt_names_test)):
            for x in range(len(Tt_names_train)):
                if Tt_names_test[y] == Tt_names_train[x]:
                    new_df['Tt' + str(x)] = df.iloc[:, 129 + len(Tag_names_test) + y]

    processed_df = pd.concat([df.iloc[:, 0:129], new_df], axis=1, sort=False)
    processed_df = processed_df.fillna(0)

    return processed_df


# y: Convert text to numeric
def ConvertNumericLabel(df):
    df_orig = df

    # df 2// movieId// genres
    vectorizer = CountVectorizer()

    df = df.iloc[:, 1]
    # [1] handle 3 NaN element
    sum = df.isnull().sum()
    print(sum)

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # strategy = 'most_frequent'
    imp = imp.fit(df.to_numpy().reshape(-1, 1))
    df = imp.transform(df.to_numpy().reshape(-1, 1)).ravel()  # Nan to constant
    df = df.reshape(-1, 1)
    # sum = df_title.isnull().sum()
    # print(sum) #Nan element->0 after processing

    # [2] Text to Number
    # arry to df
    df = pd.DataFrame(data=df[0:, 0:], index=[i for i in range(df.shape[0])],
                      columns=['f' + str(i) for i in range(df.shape[1])])
    # df to ser
    df = df.iloc[:, 0]

    C = vectorizer.fit_transform(df)
    global  convert_tb
    convert_tb = vectorizer.get_feature_names()

    #print("C", len(vectorizer.get_feature_names()), "--", vectorizer.get_feature_names())
    Processed_df = pd.DataFrame(data=C.toarray()[0:, 0:], index=[i for i in range(C.toarray().shape[0])],
                                columns=['f' + str(i) for i in range(C.toarray().shape[1])])

    #combine to one col
    for index, row in Processed_df.iterrows():
        for x in range(row.size):
            if (row[x] == 1):
                row[0] = x

    ##### Concat Numerized columns with original table
    df_orig = df_orig.drop(columns=['genres'])
    df_numeric = pd.concat([df_orig, Processed_df.iloc[:, 0]], axis=1, sort=False)
    # print(df_numeric)

    return df_numeric


# Function to perform training with giniIndex.
def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train.iloc[:, 1])
    return clf_gini


# Function to perform training with entropy.
def train_using_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train.iloc[:, 1])
    return clf_entropy


# Function to make predictions
def prediction(X_valid, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_valid)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred, baseline):
    if baseline != "true":
        print("Confusion Matrix: ",
              confusion_matrix(y_test, y_pred))

        print("Accuracy : ",
              accuracy_score(y_test, y_pred) * 100)

        print("Report : ",
              classification_report(y_test, y_pred))

    return accuracy_score(y_test, y_pred) * 100


# oneR Baseline
def baseline(X_valid, y_valid):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

    # todo 모든 feature iterate 하면서 제일 결과 좋은 feature 찾기 (Decision stump)
    max = -1;
    max_idx = 0;
    #feature_score = []

    for x in range(X_valid.iloc[0, :].size):
        # todo curr : 모든 컬럼 돌면서 fit & pred 한다음에 max acc인 컬럼으로 acc 정하는중
        # todo acc 버리지 말고 arr 담아서 feature selection으로 사용하자

        one_feature = X_valid.iloc[:, x]
        clf_gini.fit(one_feature.to_numpy().reshape(-1, 1), y_valid.iloc[:, 1])
        y_pred = clf_gini.predict(one_feature.to_numpy().reshape(-1, 1))
        acc = cal_accuracy(y_valid.iloc[:, 1].array, y_pred, "true")
        # feature_score.append(acc)

        if acc >= max:
            max = acc
            max_idx = x

    print("Max accuracy:", max)
    print("Decision Stump feature index: ", max_idx)

    return max


def compareGiniEntropy(X_train, X_valid, X_test, y_train, y_valid):
    # training classifier
    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = train_using_entropy(X_train, y_train)

    # testing classifier
    print("Results Using Baseline:")
    baseline(X_valid, y_valid) #TODO feature Selection 전으로 넣을까...???? 그래야 의미가 있음

    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_valid, clf_gini)
    # y_pred_gini = prediction(X_valid.iloc[:, 1:128], clf_gini)
    cal_accuracy(y_valid.iloc[:, 1].array, y_pred_gini, "false")

    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_valid, clf_entropy)
    cal_accuracy(y_valid.iloc[:, 1].array, y_pred_entropy, "false")

    # Need only testing phase for Kaggle w/ prediction(X_test, clf_object)


def compareMaxDepth(X_train, X_valid, X_test, y_train, y_valid):
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []

    for max_depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(X_train, y_train.iloc[:, 1])


        train_pred = dt.predict(X_train)
        ACC1 = accuracy_score(y_train.iloc[:, 1].to_numpy(), train_pred)
        train_results.append(ACC1)

        y_pred = dt.predict(X_valid)
        ACC2 =  accuracy_score(y_valid.iloc[:, 1].to_numpy(), y_pred)
        test_results.append(ACC2)  # Add auc score to previous test results

    line1, = plt.plot(max_depths, train_results, 'b', label="Train ACC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test ACC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("ACC score")
    plt.xlabel("Tree depth")
    plt.show()


def compareMinSamplesSplit(X_train, X_valid, X_test, y_train, y_valid):
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
        dt.fit(X_train, y_train.iloc[:, 1])

        train_pred = dt.predict(X_train)
        ACC1 = accuracy_score(y_train.iloc[:, 1].to_numpy(), train_pred)
        train_results.append(ACC1)

        y_pred = dt.predict(X_valid)
        ACC2 = accuracy_score(y_valid.iloc[:, 1].to_numpy(), y_pred)
        acc_count = accuracy_score(y_valid.iloc[:, 1].to_numpy(), y_pred, normalize=False)
        test_results.append(ACC2)


    line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train ACC")
    line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test ACC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("ACC score")
    plt.xlabel("Min Samples Split")
    plt.show()


def compareMinSamplesLeaf(X_train, X_valid, X_test, y_train, y_valid):
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        dt.fit(X_train, y_train.iloc[:, 1])
        train_pred = dt.predict(X_train)

        train_pred = dt.predict(X_train)
        ACC1 = accuracy_score(y_train.iloc[:, 1].to_numpy(), train_pred)
        train_results.append(ACC1)

        y_pred = dt.predict(X_valid)
        ACC2 = accuracy_score(y_valid.iloc[:, 1].to_numpy(), y_pred)
        test_results.append(ACC2)


    line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train ACC")
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test ACC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("ACC score")
    plt.xlabel("Min Samples Leaf")
    plt.show()


def compareMaxFeature(X_train, X_valid, X_test, y_train, y_valid):

    #max_features = list(range(1, X_train.shape[1]))
    max_features = np.linspace(1, X_train.shape[1], 30, endpoint=True)

    train_results = []
    test_results = []
    for max_feature in max_features:
        dt = DecisionTreeClassifier(max_features=int(max_feature))
        dt.fit(X_train, y_train)

        train_pred = dt.predict(X_train)
        ACC1 = accuracy_score(y_train.iloc[:, 1].to_numpy(), train_pred[:,1])
        train_results.append(ACC1)

        y_pred = dt.predict(X_valid)
        ACC2 = accuracy_score(y_valid.iloc[:, 1].to_numpy(), y_pred[:,1])
        test_results.append(ACC2)

    line1, = plt.plot(max_features, train_results, 'b', label="Train ACC")
    line2, = plt.plot(max_features, test_results, 'r', label="Test ACC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("ACC score")
    plt.xlabel("Max Feature")
    plt.show()


def compareKFeatureSel(X_train, X_valid, X_test, y_train, y_valid):

    #max_features = list(range(1, X_train.shape[1]))
    ks = np.linspace(1, X_train.shape[1], 30, endpoint=True)
    ks = sorted(ks, reverse=True)

    train_results = []
    test_results = []

    for k in ks:
        X_train, X_valid, X_test, y_train, y_valid = Ft_Sel(X_train, X_valid, X_test, y_train, y_valid, int(k)) #k= 1일때 X_Train 다 드랍시키고 컬럼 하나만 남음...
        dt = DecisionTreeClassifier(max_depth=16, min_samples_leaf=5)
        dt.fit(X_train, y_train.iloc[:, 1])

        train_pred = dt.predict(X_train)
        ACC1 = accuracy_score(y_train.iloc[:, 1].to_numpy(), train_pred)
        train_results.append(ACC1)

        y_pred = dt.predict(X_valid)
        ACC2 = accuracy_score(y_valid.iloc[:, 1].to_numpy(), y_pred)
        test_results.append(ACC2)

    line1, = plt.plot(ks, train_results, 'b', label="Train ACC")
    line2, = plt.plot(ks, test_results, 'r', label="Test ACC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("ACC score")
    plt.xlabel("K Best Feature Selection")
    plt.show()


def Final_DT_CF(X_train, X_valid, X_test, y_train, y_valid):
    # training classifier
    print("[Final-Training Result]:")
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=16, min_samples_split=0.1)
    clf_gini.fit(X_train, y_train.iloc[:, 1])
    y_pred = prediction(X_valid, clf_gini)
    cal_accuracy(y_valid.iloc[:, 1].array, y_pred, "false")

    print("[Final-Labeled Result]:")
    y_pred_test = prediction(X_test, clf_gini)

    new_res = []
    for num_lb in y_pred_test :
        new_res.append(convert_tb[num_lb])

    df_res = pd.DataFrame(new_res, columns=['genres'])
    print(df_res)
    df_res.to_clipboard(sep=',', index=False)


    # Need only testing phase for Kaggle w/ prediction(X_test, clf_object)



def main():
    X_train, X_valid, X_test, y_train, y_valid = importData()
    X_train, X_valid, X_test, y_train, y_valid = Ft_Sel(X_train, X_valid, X_test, y_train, y_valid)

    ##Thesis Section 1 to Section 4
    #compareGiniEntropy(X_train, X_valid, X_test, y_train, y_valid)


    #compareKFeatureSel(X_train, X_valid, X_test, y_train, y_valid)


    #compareMaxDepth(X_train, X_valid, X_test, y_train, y_valid)
    #compareMinSamplesSplit(X_train, X_valid, X_test, y_train, y_valid)
    #compareMinSamplesLeaf(X_train, X_valid, X_test, y_train, y_valid)
    #compareMaxFeature(X_train, X_valid, X_test, y_train, y_valid)

    Final_DT_CF(X_train, X_valid, X_test, y_train, y_valid)



# Calling main function
if __name__ == "__main__":
    main()



