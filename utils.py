import pandas as pd


def compare_feature_imp_corr(estimator, df, target_name):
    """
    Return a DataFrame which compares the signals' Feature
    Importance in the Machine Learning model, to the absolute
    correlation of the signals and stock-returns.

    :param estimator: Sklearn ensemble estimator.
    :return: Pandas DataFrame.
    """

    # Wrap the list of Feature Importance in a Pandas Series.
    df_feat_imp = pd.Series(estimator.feature_importances_,
                            index=df.drop(target_name, axis=1).columns,
                            name='Feature Importance')

    df_corr_returns = df.corrwith(df[target_name]).abs().sort_values(
        ascending=False).rename(f'{target_name} correlation')[1:]

    # Concatenate the DataFrames with Feature Importance
    # and Return Correlation.
    dfs = [df_feat_imp, df_corr_returns]
    df_compare = pd.concat(dfs, axis=1, sort=True)

    # Sort by Feature Importance.
    df_compare.sort_values(by='Feature Importance',
                           ascending=False, inplace=True)

    return df_compare
