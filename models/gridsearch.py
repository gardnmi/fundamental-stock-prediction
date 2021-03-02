from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def cv_splits(df, tickers):
    df_copy = df.copy()

    # Need to Create a numberical index for cv
    df_copy = df_copy.reset_index().set_index('Ticker', append=True)
    df_copy = df_copy.reorder_levels(['Ticker', None])
    df_copy = df_copy.drop('Date', axis=1)

    cv_splits = []
    for n in range(3):
        tickers_train, tickers_test = train_test_split(tickers, train_size=0.7)

        train_index = df_copy.loc[tickers_train].index.get_level_values(1)
        test_index = df_copy.loc[tickers_test].index.get_level_values(1)

        cv_splits.append((train_index, test_index))

    return cv_splits


ct = make_column_transformer(
    (OrdinalEncoder(), ['Sector', 'Industry', 'Fiscal Period']),
    remainder='passthrough'
)

# Grid Search
model = XGBRegressor()

parameters = {
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'min_child_weight': [2, 4],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.7, 0.9],
    'n_estimators': [100, 200, 400]}

grid_search = GridSearchCV(model,
                           parameters,
                           # this line can be commented to use XGB's default metric
                           scoring='neg_mean_squared_error',
                           cv=cv_index,
                           verbose=True)

grid_search.fit(ct.fit_transform(X), y)

grid_search.best_params_


sns.scatterplot(data=sf.winsorize(pred_df[['Ticker', 'Close', 'Pred Close']].groupby(
    'Ticker').mean(), clip=False, quantile=.01), x='Close', y='Pred Close')


explainer = lime.lime_tabular.LimeTabularExplainer(training_date=df.to_numpy(),
                                                   mode='regression',
                                                   feature_names=df.columns.tolist(),
                                                   class_names=['Close'],
                                                   verbose=True,
                                                   mode='regression')


[
    'Revenue ',
    'Gross Profit ',
    'Operating Income (Loss) ',
    'Net Income ',
    'Cash, Cash Equivalents & Short Te'rm Investments,
    'Accounts & Notes Receivable ',
    'Long Term Debt ',
    'Total Liabilities ',
    'Retained Earnings',
    'Change in Working Capital',
    'Net Cash from Operating Activitie's,
    'Net Cash from Investing Activitie's,
    'Cash from (Repayment of) Debt',
    'Cash from (Repurchase of) Equity',
    'Net Cash from Financing Activitie's,
    'Net Change in Cash',
    '(Dividends + Share Buyback) / FCF',
    'Asset Turnover',
    'CapEx / (Depr + Amor)',
    'Current Ratio',
    'Dividends / FCF',
    'Gross Profit Margin',
    'Interest Coverage',
    'Net Profit Margin',
    'Quick Ratio',
    'Return on Assets',
    'Return on Equity',
    'Share Buyback / FCF',
    'Assets Growth',
    'Assets Growth QOQ',
    'Assets Growth YOY',
    'Earnings Growth',
    'Earnings Growth QOQ',
    'Earnings Growth YOY',
    'FCF Growth',
    'FCF Growth QOQ',
    'FCF Growth YOY',
    'Sales Growth',
    'Sales Growth QOQ',
    'Sales Growth YOY',
    'Free Cash Flow',
    'Operating Margin',
    'Free Cash Flow to Net Income',
    'Liabilities to Equity Ratio',
    'Earnings Per Share, Basic',
    'Earnings Per Share, Diluted',
    'Sales Per Share',
    'Equity Per Share',
    'Free Cash Flow Per Share'
]
