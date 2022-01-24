This project has 2 python files and one Data folder.

[1] DecisionTrees.py
    main() function has 6 inactive functions and 1 active function as below after the function importData() and Ft_Sel().

    #compareGiniEntropy(X_train, X_valid, X_test, y_train, y_valid)
    #compareKFeatureSel(X_train, X_valid, X_test, y_train, y_valid)
    #compareMaxDepth(X_train, X_valid, X_test, y_train, y_valid)
    #compareMinSamplesSplit(X_train, X_valid, X_test, y_train, y_valid)
    #compareMinSamplesLeaf(X_train, X_valid, X_test, y_train, y_valid)
    #compareMaxFeature(X_train, X_valid, X_test, y_train, y_valid)
    Final_DT_CF(X_train, X_valid, X_test, y_train, y_valid)
    
    Each function is for the experiment/ prove the theory of the hypothesis of this assignment.
    To get results as the paper, remove the "#" and run. For more details of the codes, please find it in the paper.

[2] NaiveBayes.py
    main() function has 4 code blocks as below after the function importData(). (In this program, importData covers feature selection)

    #Baseline
    #Gaussian Naive Bayes
    #Multinomial Naive Bayes
    #Labeling Test Data