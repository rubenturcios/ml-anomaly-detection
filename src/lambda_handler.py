import asyncio
import json
import os
from pprint import pprint
from typing import Dict, List, Tuple

import boto3
import pandas as pd
from scipy.stats import percentileofscore

from utils import (
    Credentials,
    get_secret, 
    get_db_connection,
    get_save_and_exit_df,
    get_event_training_data,
    get_prediction_means_df,
    get_person_of_interest_data
)


RDS_SECRET_NAME = os.getenv('RDS_SECRET_NAME', "dev-anamoly-detection-rds")
SCHEMA = os.getenv('SCHEMA', 'public')
EVENT_PREDICTED_TABLE = os.getenv('EVENT_PREDICTED_TABLE', 'event_predicted')
PERSON_OF_INTEREST_PREDICTED_TABLE = os.getenv('PERSON_OF_INTEREST_PREDICTED_TABLE', 'person_of_interest_predicted')

EVENT_PREDICTION_ENDPOINT_NAME = os.getenv(
    'EVENT_PREDICTION_ENDPOINT_NAME',
    'anamoly-detection-event-prediction-endpoint'
)
EVENT_ESTIMATOR_ENDPOINT_NAME = os.getenv(
    'EVENT_ESTIMATOR_ENDPOINT_NAME',
    'anamoly-detection-event-estimator-endpoint'
)
PERSON_OF_INTEREST_PREDICTION_ENDPOINT_NAME = os.getenv(
    'PERSON_OF_INTEREST_PREDICTION_ENDPOINT_NAME',
    'anamoly-detection-person_of_interest-prediction-endpoint'
)
PERSON_OF_INTEREST_ESTIMATOR_ENDPOINT_NAME = os.getenv(
    'PERSON_OF_INTEREST_ESTIMATOR_ENDPOINT_NAME',
    'anamoly-detection-person_of_interest-estimator-endpoint'
)

QUERY = f'''
select h.id as event_id, 
h.facility_uuid,
h."timestamp", 
h.latitude as event_latitude,
h.longitude as event_longitude, 
h."event", 
h.altitude,
v.project_id, 
p.customer_id,
v.latitude as facility_latitude, 
v.longitude as facility_longitude, 
v.revision,
v.person_of_interest_id
from custom h 
join visit v on v.uid = h.facility_uuid 
join usr u on v.person_of_interest_id = u.id 
join project p on p.id = v.project_id
where (h."event" = 9 or h."event" = 13) 
and "timestamp" > now() - {os.getenv("INTERVAL", "interval '7 day'")} 
order by h."timestamp" desc
'''

session = boto3.Session()
sagemaker_client = session.client('sagemaker-runtime')


def get_credentials(secret_name: str) -> Credentials:
    secret = json.loads(get_secret(secret_name))
    return Credentials(
        USERNAME=secret['username'],
        PASSWORD=secret['password'],
        HOST=secret['host'],
        DB=secret['dbname'],
        PORT=secret['port']
    )


def get_prediction(endpoint_name: str, data: pd.DataFrame) -> dict:
    return json.loads(sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=data.to_csv(header=False, index=False).encode(),
        ContentType='text/csv'
    )['Body'].read().decode())


def get_score_and_percentile(scores: Dict[str, float]) -> Tuple[List[float], List[float]]:
    raw_score = scores['predicted_decision_scores']
    reference_scores = scores['fitted_decision_scores']
    percentiles = percentileofscore(reference_scores, raw_score)
    return raw_score, percentiles


def get_event_predictions(df_save_and_exit_only: pd.DataFrame) -> pd.DataFrame:
    event_model_data = get_event_training_data(df_save_and_exit_only)
    event_model_response = get_prediction(EVENT_PREDICTION_ENDPOINT_NAME, event_model_data)
    raw_scores, percentiles = get_score_and_percentile(event_model_response)
    df_save_and_exit_only['raw_score'] = raw_scores
    df_save_and_exit_only['score_percentile'] = percentiles
    shap_values = get_shap_values(EVENT_ESTIMATOR_ENDPOINT_NAME, event_model_data)
    return pd.concat([df_save_and_exit_only, shap_values], axis=1)


def get_person_of_interest_predictions(df_save_and_exit_only: pd.DataFrame) -> pd.DataFrame:
    df_with_means = get_prediction_means_df(df_save_and_exit_only)
    person_of_interest_model_data = get_person_of_interest_data(df_with_means)
    person_of_interest_model_response = get_prediction(PERSON_OF_INTEREST_PREDICTION_ENDPOINT_NAME, person_of_interest_model_data)
    raw_scores, percentiles = get_score_and_percentile(person_of_interest_model_response)
    df_with_means['raw_score'] = raw_scores
    df_with_means['confidence_rank'] = 100 - percentiles
    df_with_means['date_predicted'] = pd.Timestamp.now()
    shap_values = get_shap_values(PERSON_OF_INTEREST_ESTIMATOR_ENDPOINT_NAME, person_of_interest_model_data)
    return pd.concat([df_with_means, shap_values], axis=1)


def get_shap_values(endpoint_name: str, data: pd.DataFrame, size: int = 1000) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        asyncio.gather(
            *[
                loop.run_in_executor(
                    None,
                    get_prediction,
                    endpoint_name,
                    data.iloc[i:i + size]
                ) for i in range(0, len(data), size)
            ]
        )
    )
    return pd.concat(
        (pd.DataFrame.from_dict(result) for result in results)
    ).reset_index(drop=True)


def handler(event: dict, context: dict):
    credentials = get_credentials(RDS_SECRET_NAME)
    db_connection = get_db_connection(credentials)
    df = pd.read_sql(QUERY, db_connection)
    print('Pulled Data...')

    df_save_and_exit_only = get_save_and_exit_df(df)
    event_predictions = get_event_predictions(df_save_and_exit_only)
    person_of_interest_predictions = get_person_of_interest_predictions(df_save_and_exit_only)

    event_predicted_response_code = event_predictions.to_sql(
        EVENT_PREDICTED_TABLE,
        db_connection,
        schema=SCHEMA,
        index=False,
        if_exists='append'
    )
    person_of_interest_predicted_response_code = person_of_interest_predictions.to_sql(
        PERSON_OF_INTEREST_PREDICTED_TABLE,
        db_connection,
        schema=SCHEMA,
        index=False,
        if_exists='append'
    )
    db_connection.commit()
