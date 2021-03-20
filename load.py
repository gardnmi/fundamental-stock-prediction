import pandas as pd
import numpy as np
import simfin as sf
from simfin.names import *
import pathlib
from category_encoders import OrdinalEncoder

DATA_DIR = pathlib.Path('./data')


def load_shareprices(
        refresh_days=1,
        simfin_api_key='free',
        simfin_directory='simfin_data/'):

    # Set Simfin Settings
    sf.set_api_key(simfin_api_key)
    sf.set_data_dir(simfin_directory)

    # Used by all datasets
    shareprices_df = sf.load_shareprices(
        variant='daily', market='us', refresh_days=refresh_days)

    # Merge Fundamental with Stock Prices
    # Downsample Share Prices to Rolling 30 Day End of Month
    shareprices_df = shareprices_df[['Close']].groupby('Ticker').rolling(
        30, min_periods=1).mean().reset_index(0, drop=True)
    shareprices_df = sf.resample(
        df=shareprices_df, rule='M', method=lambda x: x.last())

    return shareprices_df


def load_dataset(refresh_days=1,
                 dataset='general',
                 thresh=0.7,
                 simfin_api_key='free',
                 simfin_directory='simfin_data/',
                 data_directory=DATA_DIR,
                 shareprices_df=''
                 ):

    # Set Simfin Settings
    sf.set_api_key(simfin_api_key)
    sf.set_data_dir(simfin_directory)

    derived_shareprice_df = sf.load_derived_shareprices(
        variant='latest', market='us')
    derived_shareprice_df.to_csv(data_directory/'stock_derived.csv')

    company_df = sf.load_companies(market='us', refresh_days=1)
    company_df.to_csv(data_directory/'company.csv')

    industry_df = sf.load_industries(refresh_days=1)
    industry_df.to_csv(data_directory/'industry.csv')

    if dataset == 'general':

        # Load Data from Simfin
        income_df = sf.load_income(
            variant='ttm', market='us', refresh_days=refresh_days)
        income_df = income_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        income_quarterly_df = sf.load_income(
            variant='quarterly', market='us', refresh_days=refresh_days)
        income_quarterly_df = income_quarterly_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        income_df.groupby('Ticker').last().to_csv(
            data_directory/'general_income.csv')

        balance_df = sf.load_balance(
            variant='ttm', market='us', refresh_days=refresh_days)
        balance_df = balance_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        balance_quarterly_df = sf.load_balance(
            variant='quarterly', market='us', refresh_days=refresh_days)
        balance_quarterly_df = balance_quarterly_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        balance_df.groupby('Ticker').last().to_csv(
            data_directory/'general_balance.csv')

        cashflow_df = sf.load_cashflow(
            variant='ttm', market='us', refresh_days=refresh_days)
        cashflow_df = cashflow_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        cashflow_quarterlay_df = sf.load_cashflow(
            variant='quarterly', market='us', refresh_days=refresh_days)
        cashflow_quarterlay_df = cashflow_quarterlay_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        cashflow_df.groupby('Ticker').last().to_csv(
            data_directory/'general_cashflow.csv')

        derived_df = sf.load_derived(
            variant='ttm', market='us', refresh_days=refresh_days)
        derived_df = derived_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        derived_df.groupby('Ticker').last().to_csv(
            data_directory/'general_fundamental_derived.csv')

        cache_args = {'cache_name': 'financial_signals',
                      'cache_refresh': refresh_days}

        fin_signal_df = sf.fin_signals(df_income_ttm=income_df,
                                       df_balance_ttm=balance_df,
                                       df_cashflow_ttm=cashflow_df,
                                       **cache_args)

        growth_signal_df = sf.growth_signals(df_income_ttm=income_df,
                                             df_income_qrt=income_quarterly_df,
                                             df_balance_ttm=balance_df,
                                             df_balance_qrt=balance_quarterly_df,
                                             df_cashflow_ttm=cashflow_df,
                                             df_cashflow_qrt=cashflow_quarterlay_df,
                                             **cache_args)

        # Remove Columns that exist in other Fundamental DataFrames
        balance_columns = balance_df.columns[~balance_df.columns.isin(
            set().union(income_df.columns))]
        cashflow_columns = cashflow_df.columns[~cashflow_df.columns.isin(
            set().union(income_df.columns))]
        derived_df_columns = derived_df.columns[~derived_df.columns.isin(set().union(income_df.columns,
                                                                                     growth_signal_df.columns,
                                                                                     fin_signal_df.columns))]

        # Merge the fundamental data into a single dataframe
        fundamental_df = income_df.join(balance_df[balance_columns]
                                        ).join(cashflow_df[cashflow_columns]
                                               ).join(fin_signal_df
                                                      ).join(growth_signal_df
                                                             ).join(derived_df[derived_df_columns])

        fundamental_df['Dataset'] = 'general'

    elif dataset == 'banks':

        # Load Data from Simfin
        income_df = sf.load_income_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        income_df = income_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        income_df.groupby('Ticker').last().to_csv(
            data_directory/'banks_income.csv')

        balance_df = sf.load_balance_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        balance_df = balance_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        balance_df.groupby('Ticker').last().to_csv(
            data_directory/'banks_balance.csv')

        cashflow_df = sf.load_cashflow_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        cashflow_df = cashflow_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        cashflow_df.groupby('Ticker').last().to_csv(
            data_directory/'banks_cashflow.csv')

        derived_df = sf.load_derived_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        derived_df = derived_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        derived_df.groupby('Ticker').last().to_csv(
            data_directory/'banks_fundamental_derived.csv')
        derived_df.groupby('Ticker').last().to_csv(
            data_directory/'banks_fundamental_derived.csv')

        # Remove Columns that exist in other Fundamental DataFrames
        balance_columns = balance_df.columns[~balance_df.columns.isin(
            set().union(income_df.columns))]
        cashflow_columns = cashflow_df.columns[~cashflow_df.columns.isin(
            set().union(income_df.columns))]
        derived_df_columns = derived_df.columns[~derived_df.columns.isin(
            set().union(income_df.columns))]

        # Merge the fundamental data into a single dataframe
        fundamental_df = income_df.join(balance_df[balance_columns]
                                        ).join(cashflow_df[cashflow_columns]
                                               ).join(derived_df[derived_df_columns])

        fundamental_df['Dataset'] = 'banks'

    elif dataset == 'insurance':

        # Load Data from Simfin
        income_df = sf.load_income_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)
        income_df = income_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        income_df.groupby('Ticker').last().to_csv(
            data_directory/'insurance_income.csv')

        balance_df = sf.load_balance_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)
        balance_df = balance_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        balance_df.groupby('Ticker').last().to_csv(
            data_directory/'insurance_balance.csv')

        cashflow_df = sf.load_cashflow_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)
        cashflow_df = cashflow_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        cashflow_df.groupby('Ticker').last().to_csv(
            data_directory/'insurance_cashflow.csv')

        derived_df = sf.load_derived_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)
        derived_df = derived_df.sort_index(
            level=['Ticker', 'Report Date'], ascending=[1, 1])
        derived_df.groupby('Ticker').last().to_csv(
            data_directory/'insurance_fundamental_derived.csv')

        # Remove Columns that exist in other Fundamental DataFrames
        balance_columns = balance_df.columns[~balance_df.columns.isin(
            set().union(income_df.columns))]
        cashflow_columns = cashflow_df.columns[~cashflow_df.columns.isin(
            set().union(income_df.columns))]
        derived_df_columns = derived_df.columns[~derived_df.columns.isin(
            set().union(income_df.columns))]

        # Merge the fundamental data into a single dataframe
        fundamental_df = income_df.join(balance_df[balance_columns]
                                        ).join(cashflow_df[cashflow_columns]
                                               ).join(derived_df[derived_df_columns])

        fundamental_df['Dataset'] = 'insurance'

    # Drop Columns with more then 1-thresh nan values
    fundamental_df = fundamental_df.dropna(
        thresh=int(thresh*len(fundamental_df)), axis=1)

    # Drop Duplicate Index
    fundamental_df = fundamental_df[~fundamental_df.index.duplicated(
        keep='first')]

    # Replace Report Date with the Publish Date because the Publish Date is when the Fundamentals are known to the Public
    fundamental_df['Published Date'] = fundamental_df['Publish Date']
    fundamental_df = fundamental_df.reset_index(
    ).set_index(['Ticker', 'Publish Date'])

    df = sf.reindex(df_src=fundamental_df, df_target=shareprices_df, group_index=TICKER, method='ffill'
                    ).dropna(how='all').join(shareprices_df)

    # General
    # Clean Up
    df = df.drop(['SimFinId', 'Currency', 'Fiscal Year', 'Report Date',
                  'Restated Date', 'Fiscal Period', 'Published Date'], axis=1)

    if dataset == 'general':
        # Remove Share Prices Over Amazon Share Price
        df = df[df['Close'] <= df.loc['AMZN']['Close'].max()]

        df = df.dropna(
            subset=['Shares (Basic)', 'Shares (Diluted)', 'Revenue', 'Earnings Growth'])

        non_per_share_cols = ['Currency', 'Fiscal Year', 'Fiscal Period', 'Published Date',
                              'Restated Date', 'Shares (Basic)', 'Shares (Diluted)', 'Close', 'Dataset'
                              ] + fin_signal_df.columns.tolist() + growth_signal_df.columns.tolist() + derived_df_columns.difference(['EBITDA', 'Total Debt', 'Free Cash Flow']).tolist()

    else:
        df = df.dropna(
            subset=['Shares (Basic)', 'Shares (Diluted)', 'Revenue'])

        non_per_share_cols = ['Currency', 'Fiscal Year', 'Fiscal Period', 'Published Date',
                              'Restated Date', 'Shares (Basic)', 'Shares (Diluted)', 'Close', 'Dataset'
                              ] + derived_df_columns.difference(['EBITDA', 'Total Debt', 'Free Cash Flow']).tolist()

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    per_share_cols = df.columns[~df.columns.isin(non_per_share_cols)]

    df[per_share_cols] = df[per_share_cols].div(
        df['Shares (Diluted)'], axis=0)

    # Add Company and Industry Information and Categorize
    df = df.join(company_df).merge(industry_df, left_on='IndustryId', right_index=True).drop(
        columns=['IndustryId', 'Company Name', 'SimFinId'])

    categorical_features = [
        col for col in df.columns if df[col].dtype == 'object']

    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(df)

    df = encoder.transform(df)

    # Sort
    df = df.sort_index(level=['Ticker', 'Date'], ascending=[1, 1])

    return df
