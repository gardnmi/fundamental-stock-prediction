# Move the Update Data Logic Here
import simfin as sf
from load import load_dataset, load_shareprices
import pathlib
import os
from dotenv import load_dotenv
from predict import train, predict, predict_similiar

load_dotenv()
SIMFIN_API_KEY = os.getenv('SIMFIN_API_KEY', 'free')
MODELS_DIR = pathlib.Path('./models')
DATA_DIR = pathlib.Path('./data')

# LOAD
shareprices_df = load_shareprices(simfin_api_key=SIMFIN_API_KEY)
general_df = load_dataset(
    dataset='general', simfin_api_key=SIMFIN_API_KEY, shareprices_df=shareprices_df)
banks_df = load_dataset(
    dataset='banks', simfin_api_key=SIMFIN_API_KEY, shareprices_df=shareprices_df)
insurance_df = load_dataset(
    dataset='insurance', simfin_api_key=SIMFIN_API_KEY, shareprices_df=shareprices_df)

# TRAIN
general_model = train(general_df,
                      winsor_quantile=0.01,
                      model_name='general_model',
                      feature_name='general',
                      param=dict(learning_rate=0.01,
                                 max_depth=3,
                                 subsample=.5,
                                 colsample_bylevel=0.7,
                                 colsample_bytree=0.7,
                                 n_estimators=200))

banks_model = train(banks_df,
                    winsor_quantile=0.05,
                    model_name='banks_model',
                    feature_name='banks',
                    param=dict(learning_rate=0.01,
                               max_depth=2,
                               subsample=.8,
                               colsample_bylevel=0.7,
                               colsample_bytree=0.7,
                               n_estimators=200))

insurance_model = train(insurance_df,
                        winsor_quantile=0.08,
                        model_name='insurance_model',
                        feature_name='insurance',
                        param=dict(learning_rate=0.01,
                                   max_depth=2,
                                   subsample=1,
                                   colsample_bylevel=0.7,
                                   colsample_bytree=0.7,
                                   n_estimators=150))

# PREDICT
general_df = predict(general_model, general_df, 'general_predictions')
banks_df = predict(banks_model, banks_df, 'banks_predictions')
insurance_df = predict(insurance_model, insurance_df, 'insurance_predictions')

# PREDICT SIMILIAR STOCKS
general_matrix_df = predict_similiar(
    general_model, general_df, 'general_sim_matrix')

banks_matrix_df = predict_similiar(banks_model, banks_df, 'banks_sim_matrix')

insurance_matrix_df = predict_similiar(
    insurance_model, insurance_df, 'insurance_sim_matrix')
