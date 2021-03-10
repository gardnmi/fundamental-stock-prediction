import pandas as pd
import numpy as np
import simfin as sf
from simfin.names import *


def load_dataset(refresh_days=1, dataset='common', thresh=0.7, simfin_api_key='free', simfin_directory='simfin_data/'):

    # Set Simfin Settings
    sf.set_api_key(simfin_api_key)
    sf.set_data_dir(simfin_directory)

    # Used by all datasets
    shareprices_df = sf.load_shareprices(
        variant='daily', market='us', refresh_days=refresh_days)
    company_df = sf.load_companies(market='us', refresh_days=refresh_days)
    industry_df = sf.load_industries(refresh_days=refresh_days)

    if dataset == 'common':

        # Load Data from Simfin
        income_df = sf.load_income(
            variant='ttm', market='us', refresh_days=refresh_days)
        income_quarterly_df = sf.load_income(
            variant='quarterly', market='us', refresh_days=refresh_days)

        balance_df = sf.load_balance(
            variant='ttm', market='us', refresh_days=refresh_days)
        balance_quarterly_df = sf.load_balance(
            variant='quarterly', market='us', refresh_days=refresh_days)

        cashflow_df = sf.load_cashflow(
            variant='ttm', market='us', refresh_days=refresh_days)
        cashflow_quarterlay_df = sf.load_cashflow(
            variant='quarterly', market='us', refresh_days=refresh_days)

        derived_df = sf.load_derived(
            variant='ttm', market='us', refresh_days=refresh_days)

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

        fundamental_df['Dataset'] = 'common'

    elif dataset == 'banks':

        # Load Data from Simfin
        income_df = sf.load_income_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        balance_df = sf.load_balance_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        cashflow_df = sf.load_cashflow_banks(
            variant='ttm', market='us', refresh_days=refresh_days)
        derived_df = sf.load_derived_banks(
            variant='ttm', market='us', refresh_days=refresh_days)

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
        balance_df = sf.load_balance_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)
        cashflow_df = sf.load_cashflow_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)
        derived_df = sf.load_derived_insurance(
            variant='ttm', market='us', refresh_days=refresh_days)

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

    # Merge Fundamental with Stock Prices
    # Downsample share prices to monthly
    shareprices_df = sf.resample(
        df=shareprices_df[['Close']], rule='M', method='mean')

    df = sf.reindex(df_src=fundamental_df, df_target=shareprices_df, group_index=TICKER, method='ffill'
                    ).dropna(how='all').join(shareprices_df)

    # Common
    # Clean Up
    df = df.drop(['SimFinId', 'Currency', 'Fiscal Year', 'Report Date',
                  'Restated Date', 'Fiscal Period', 'Published Date'], axis=1)

    if dataset == 'common':
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

    return df