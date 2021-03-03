def calculate_tStatistic(df, target, numeric_cols):

    """
    Calculate the t-test on TWO RELATED samples of scores, a and b.

    This is a two-sided test for the null hypothesis that 2 related or 
    repeated samples have identical average (expected) values.
    """

    # create an empty dictionary
    t_test_results = {}
    # loop over column_list and execute code 
    for column in numeric_cols:
        group1 = df.where(df[target] == 0).dropna()[column]
        group2 = df.where(df[target] == 1).dropna()[column]
        # add the output to the dictionary 
        t_test_results[column] = stats.ttest_ind(group1,group2)
    results_df = pd.DataFrame.from_dict(t_test_results,orient='Index')
    results_df.columns = ['t-statistic','t_stat_pval']

    results_df.reset_index(inplace=True)
    results_df.rename(columns = {"index":"x"}, inplace=True)

    results_df.drop('t-statistic', axis=1, inplace=True)

    return results_df
