#NB for ASM2

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

global Tag_names_train, Tag_names_valid, Tag_names_test
global Tt_names_train, Tt_names_valid, Tt_names_test
global convert_tb

# Importing Dataset
def importData():
    X_train = pd.read_csv('./Data/train_features.tsv', sep='\t')
    y_train = pd.read_csv('./Data/train_labels.tsv', sep='\t')
    X_valid = pd.read_csv('./Data/valid_features.tsv', sep='\t')
    X_test = pd.read_csv('./Data/NEW_test_features.tsv', sep='\t')
    y_valid = pd.read_csv('./Data/valid_labels.tsv', sep='\t')

    #ConvertNumeric (Text to Numeric vals)
    X_train, X_orig_train, X_Tag_df, X_Title_df = ConvertNumeric(X_train, "train")
    X_valid, X_orig_valid, X_Tag_df_valid, X_Title_df_valid = ConvertNumeric(X_valid, "valid")
    X_test, X_orig_test, X_Tag_df_test, X_Title_df_test = ConvertNumeric(X_test, "test")
    y_train = ConvertNumericLabel(y_train)  # df 2// movieId// genres
    y_valid = ConvertNumericLabel(y_valid)

    #ProcessFeatureLength
    X_valid = ProcessFeatureLength(X_valid, "valid")
    X_test = ProcessFeatureLength(X_test, "test")

    #FeatureSelection
    X_train = preprocessing.MinMaxScaler().fit_transform(X_train) #normalize data to [0,1]  ## (-) value <= chi2 이용 못함
    X_train, sel_cols = FeatureSelection_1(X_train, y_train)
        #다 드랍 하고 sel_cols 에 있는 애들만 남기기
    X_valid = preprocessing.MinMaxScaler().fit_transform(X_valid)  # normalize data to [0,1]  ## - value <= chi2 이용 못함
    X_valid = FeatureSelection_Test(X_valid, sel_cols)
    X_test = preprocessing.MinMaxScaler().fit_transform(X_test)  # normalize data to [0,1]  ## - value <= chi2 이용 못함
    X_test = FeatureSelection_Test(X_test, sel_cols)
        #arr to df
    X_train = pd.DataFrame(data=X_train[0:, 0:], index=[i for i in range(X_train.shape[0])], columns=[i for i in range(X_train.shape[1])])
    X_valid = pd.DataFrame(data=X_valid[0:, 0:], index=[i for i in range(X_valid.shape[0])], columns=[i for i in range(X_valid.shape[1])])
    X_test = pd.DataFrame(data=X_test[0:, 0:], index=[i for i in range(X_test.shape[0])], columns=[i for i in range(X_test.shape[1])])

    return X_train, X_valid, X_test, y_train, y_valid

# X: Convert string data to numeric values
def ConvertNumeric(df, case):
    #print("ConvertNumeric: Convert string data to numeric values")
    # Year Column
    # Tag Column
    df_tag = df.iloc[:, 4]
    df_tag = df_tag.str.replace(',', ' ')
    # Title Column
    df_title = df.iloc[:, 1]
    df_title = df_title.str.replace(',', '');    df_title = df_title.str.replace(')', '');    df_title = df_title.str.replace('(', '');    df_title = df_title.str.replace(':', '');    df_title = df_title.str.replace('&', '');

    ######Tag#######################################
    #sum = df_tag.isnull().sum()
    #print("NaN value(Tag): ", sum)

    vectorizer = CountVectorizer()
    A = vectorizer.fit_transform(df_tag)
    #print("Tags" , len(vectorizer.get_feature_names()), "--", vectorizer.get_feature_names())

    #for later use in ProcessFeature()
    global Tag_names_train, Tag_names_valid, Tag_names_test
    if(case == "train") : Tag_names_train = vectorizer.get_feature_names()
    elif(case == "valid") : Tag_names_valid = vectorizer.get_feature_names()
    elif(case == "test") : Tag_names_test = vectorizer.get_feature_names()

    Tag_df = pd.DataFrame(data=A.toarray()[0:, 0:], index=[i for i in range(A.toarray().shape[0])], columns=['Tg' + str(i) for i in range(A.toarray().shape[1])])

    ######Title#######################################

    # [1] handle 3 NaN element
    #sum = df_title.isnull().sum()
    #print("NaN value(Title): ", sum) #handle 3 NaN element

    imp = SimpleImputer(missing_values=np.nan, strategy='constant') #strategy = 'most_frequent'
    imp = imp.fit(df_title.to_numpy().reshape(-1,1))
    df_title = imp.transform(df_title.to_numpy().reshape(-1,1)).ravel() #Nan to constant
    df_title = df_title.reshape(-1,1)
    #sum = df_title.isnull().sum()
    #print(sum) #Nan element->0 after processing

    # [2] Text to Number
    #arry to df
    df_title = pd.DataFrame(data=df_title[0:, 0:], index=[i for i in range(df_title.shape[0])], columns=['Tt' + str(i) for i in range(df_title.shape[1])])
    #df to ser
    df_title = df_title.iloc[:,0]

    B = vectorizer.fit_transform(df_title)
    #print("Titles", len(vectorizer.get_feature_names()), "--", vectorizer.get_feature_names())
    Title_df = pd.DataFrame(data=B.toarray()[0:, 0:], index=[i for i in range(B.toarray().shape[0])],
                            columns=['Tt' + str(i) for i in range(B.toarray().shape[1])])

    #for later use in ProcessFeature()
    global Tt_names_train, Tt_names_valid, Tt_names_test
    if(case == "train") : Tt_names_train = vectorizer.get_feature_names()
    elif(case == "valid") : Tt_names_valid = vectorizer.get_feature_names()
    elif(case == "test") : Tt_names_test = vectorizer.get_feature_names()

    df_orig = df.drop(columns=['title', 'YTId', 'tag'])

    #[3] Year str >> float
    err_idx = []
    for x in range(df_orig.iloc[:, 1].size):
        try:
            df_orig.iloc[:, 1].at[x] = float(df_orig.iloc[:, 1].at[x])
        except ValueError:
            err_idx.append(x)
            df_orig.iloc[:, 1].at[x] = float(0)
        else :
            df_orig.iloc[:, 1].at[x] = float(df_orig.iloc[:, 1].at[x])
    # missing year val << avg value assign
    avg = df_orig.iloc[:, 1].mean()
    for idx in err_idx:
        df_orig.iloc[:, 1].at[idx] = avg

    #[4] Concat Numerized columns with original table
    df_numeric = pd.concat([df_orig, Tag_df, Title_df], axis=1, sort=False)

    return df_numeric, df_orig, Tag_df, Title_df

def FeatureSelection_1(X_train, y_train):
    kb = SelectKBest(chi2, k=600)
    X_train = kb.fit_transform(X_train, y_train.iloc[:,1])
    selected_ft = kb.get_support()

    sel_cols = []
    i = 0
    for bool in selected_ft:
        if bool:
            sel_cols.append(i)
        i += 1

    return X_train, sel_cols

#Test, Valid Data
def FeatureSelection_Test(Xset, sel_cols): #ndarr

    new_arr = np.zeros((Xset.shape[0],sel_cols.__len__()))

    i = 0
    for col in sel_cols:
        new_arr[:, i] = Xset[:, col]
        i += 1

    return new_arr

#
def ProcessFeatureLength(df, case):
    global Tag_names_train, Tag_names_valid, Tag_names_test
    global Tt_names_train, Tt_names_valid, Tt_names_test
    size_Tg = len(Tag_names_train) #list
    size_Tt = len(Tt_names_train)

    # new df 생성    df element num * len
    #df.shape[0] #299
    #size #207
    cols =[]

    #train data set feature size
    for num in range(size_Tg):
        cols.append("Tg"+str(num))
    for num in range(size_Tt):
        cols.append("Tt"+str(num))
    new_df = pd.DataFrame(columns=cols)

    # Process valid data set to train set size
    if(case == "valid") :
        #Tags
        for y in range(len(Tag_names_valid)):
            for x in range(len(Tag_names_train)):
                if Tag_names_valid[y] == Tag_names_train[x] :
                    new_df['Tg'+str(x)] = df.iloc[:,129+y]
        #Titles
        for y in range(len(Tt_names_valid)):
            for x in range(len(Tt_names_train)):
                if Tt_names_valid[y] == Tt_names_train[x] :
                    new_df['Tt'+str(x)] = df.iloc[:, 129+len(Tag_names_valid)+y]

    # Process test data set to train set size
    if(case == "test"):
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
    #print(sum)

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #strategy = 'most_frequent'
    imp = imp.fit(df.to_numpy().reshape(-1,1))
    df = imp.transform(df.to_numpy().reshape(-1,1)).ravel() #Nan to constant
    df = df.reshape(-1,1)
    #sum = df_title.isnull().sum()
    #print(sum) #Nan element->0 after processing

    # [2] Text to Number
    #arry to df
    df = pd.DataFrame(data=df[0:, 0:], index=[i for i in range(df.shape[0])],
                            columns=['f' + str(i) for i in range(df.shape[1])])
    #df to ser
    df = df.iloc[:,0]

    C = vectorizer.fit_transform(df)
    global  convert_tb
    convert_tb = vectorizer.get_feature_names()

    #print("C", len(vectorizer.get_feature_names()), "--", vectorizer.get_feature_names())
    Processed_df = pd.DataFrame(data=C.toarray()[0:, 0:], index=[i for i in range(C.toarray().shape[0])],
                            columns=['f' + str(i) for i in range(C.toarray().shape[1])])

    for index, row in Processed_df.iterrows():
        for x in range(row.size):
            if(row[x] == 1):
                row[0] = x

    ##### Concat Numerized columns with original table
    df_orig = df_orig.drop(columns=['genres'])
    df_numeric = pd.concat([df_orig, Processed_df.iloc[:, 0]], axis=1, sort=False)
    #print(df_numeric)

    return df_numeric


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

    max = -1;    max_idx=0;
    for x in range(X_valid.iloc[0, :].size):
        one_feature = X_valid.iloc[:, x]
        clf_gini.fit(one_feature.to_numpy().reshape(-1,1), y_valid.iloc[:, 1])
        y_pred = clf_gini.predict(one_feature.to_numpy().reshape(-1,1))
        acc = cal_accuracy(y_valid.iloc[:,1].array, y_pred, "true")
        if acc >= max:
            max = acc
            max_idx = x

    print("Max accuracy:", max)
    print("Decision Stump feature index: ", max_idx)
    return max


def Final_DT_CF(X_train, X_valid, X_test, y_train, y_valid):
    print("[Final-Labeled Result]:")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train.iloc[:, 1])
    y_pred = mnb.predict(X_test)


    new_res = []
    for num_lb in y_pred :
        new_res.append(convert_tb[num_lb])

    df_res = pd.DataFrame(new_res, columns=['genres'])
    print(df_res)
    df_res.to_clipboard(sep=',', index=False)


def main():
    X_train, X_valid, X_test, y_train, y_valid = importData()

    #Baseline
    print("[Baseline]:")
    baseline(X_valid,y_valid)

    #Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train,y_train.iloc[:, 1])
    y_pred_gnb = gnb.predict(X_valid)
    acc = cal_accuracy(y_valid.iloc[:, 1].array, y_pred_gnb, "true")
    print("Gaussian : ", acc)


    #Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(X_train,y_train.iloc[:, 1])
    print(mnb.class_prior)


    y_pred = mnb.predict(X_valid)
    acc = cal_accuracy(y_valid.iloc[:,1].array, y_pred, "true")
    print("Multinomial : ",acc)


    #Labeling Test Data
    Final_DT_CF(X_train, X_valid, X_test, y_train, y_valid)



# Calling main function
if __name__ == "__main__":
    main()
