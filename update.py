# Move the Update Data Logic Here
import simfin as sf
from load import load_dataset
import pathlib
import os
from dotenv import load_dotenv
from predict import train, predict, predict_similiar

load_dotenv()
SIMFIN_API_KEY = os.getenv('SIMFIN_API_KEY', 'free')
MODELS_DIR = pathlib.Path('./models')
DATA_DIR = pathlib.Path('./data')

# LOAD
common_df = load_dataset(dataset='common', simfin_api_key=SIMFIN_API_KEY)
banks_df = load_dataset(dataset='banks', simfin_api_key=SIMFIN_API_KEY)
insurance_df = load_dataset(dataset='insurance', simfin_api_key=SIMFIN_API_KEY)

# TRAIN
common_model = train(common_df,
                     winsor_quantile=0.01,
                     model_name='common_model',
                     feature_name='common',
                     param=dict(learning_rate=0.01,
                                max_depth=2,
                                subsample=.5,
                                colsample_bylevel=0.7,
                                colsample_bytree=0.7,
                                n_estimators=210))

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
common_df = predict(common_model, common_df, 'common_predictions')
banks_df = predict(banks_model, banks_df, 'banks_predictions')
insurance_df = predict(insurance_model, insurance_df, 'insurance_predictions')

# PREDICT SIMILIAR STOCKS
common_matrix_df = predict_similiar(
    common_model, common_df, 'common_sim_matrix')

banks_matrix_df = predict_similiar(banks_model, banks_df, 'banks_sim_matrix')

insurance_matrix_df = predict_similiar(
    insurance_model, insurance_df, 'insurance_sim_matrix')
