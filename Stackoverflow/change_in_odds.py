def change_in_odds(X, y):
    """
    Calculates Change in odds(LogisticRegression)

    Params
    ==============================
    X          : features
    y          : target

    Return
    ===============================
    uni_log_df : A dataframe with features and "change_in_odd"
    """

    # Logistic Coefficients
    logreg = LogisticRegression(max_iter=1000)

    uni_logreg = {}

    for i in X.columns:

        # Normalization
        # X_norm = StandardScaler().fit_transform(leads_c[i])
        logreg.fit(X[[i]].values, y.values)
        coef = logreg.coef_
        # print(coef[0][0])

        exp_coef = math.exp(coef[0][0])

        uni_logreg[i] = exp_coef

    uni_log_df = pd.DataFrame(uni_logreg.items(), columns=['x', 'Change_in_Odds(LogReg)'])
    
    return uni_log_df
