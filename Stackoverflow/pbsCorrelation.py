def pbsCorrelation(df, target, numeric_cols):

    pbsc_dict = {}

    for i in numeric_cols:
        pbsc_val = list(stats.pointbiserialr(df[i].values, df[target].values))
        pbsc_dict[i] = pbsc_val
    pbsc_df =pd.DataFrame.from_dict(pbsc_dict, orient="index", columns=["PBS_Corr", "pval_pbs"])
    pbsc_df.reset_index(inplace=True)
    pbsc_df.rename(columns={"index": "x"}, inplace=True)
    pbsc_df.drop("pval_pbs", axis=1, inplace=True)

    # pbsc_df.round(decimals=4)
    return pbsc_df
