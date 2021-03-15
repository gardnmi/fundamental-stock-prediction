import pandas as pd
import seaborn as sns
import simfin as sf
import streamlit as st


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


def plot_correlation(model, X, y):

    model.fit(X, y)
    y_pred = model.predict(X)

    pred_df = pd.concat([y.reset_index(),
                         pd.Series(y_pred, name='Predicted Close')
                         ], axis=1)

    g = sns.scatterplot(data=sf.winsorize(pred_df[['Ticker', 'Close', 'Predicted Close']].groupby(
        'Ticker').mean(), clip=False, quantile=.005), x='Close', y='Predicted Close')

    text = (
        f"Correlation: {pred_df[['Ticker', 'Close', 'Predicted Close']].groupby('Ticker').mean().corr().values[0][1]:.2%}")

    g.set_title(text)


class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar.radio(
        app = st.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
