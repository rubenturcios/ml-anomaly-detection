import json
import os
import tarfile
from argparse import ArgumentParser, Namespace
from typing import Tuple, Union
from enum import Enum
import logging
from functools import partial

import joblib
import pandas as pd
from pyod.models.iforest import IForest
from sagemaker.sklearn import SKLearnModel
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from shap.explainers import TreeExplainer

from utils import (
    Credentials,
    get_db_connection,
    get_save_and_exit_df,
    get_event_training_data,
    get_means_df,
    get_person_of_interest_data,
    get_secret,
    upload_to_s3
)


MODEL_FOLDER = os.getenv('MODEL_FOLDER', './model')
DATA_FOLDER = os.getenv('DATA_PATH', './data')
ROLE = os.getenv('SAGEMAKER_ROLE', 'arn:aws:iam::1234567890:role/MlStack-SageMakerRole')
ENDPOINT_NAME_PREFIX = os.getenv('ENDPOINT_NAME_PREFIX', 'anamoly-detection')
RDS_SECRET_NAME = os.getenv('RDS_SECRET_NAME', 'dev-anamoly-detection-rds')

QUERY = """
select h.id as event_id, h.facility_uuid, v.uid as visit_facility_uuid, h."timestamp", h.latitude as event_latitude,
h.longitude as event_longitude, h."event", h.altitude,
v.project_id, v.latitude as facility_latitude, v.longitude as facility_longitude, v.revision,
v.inspection_document, v.person_of_interest_id
from custom h join visit v on v.uid = h.facility_uuid join usr u on v.person_of_interest_id = u.id where h."event" = 9 or h."event" = 13 order by h."timestamp" desc
"""

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))


class Classification(Enum):
    EVENT = 'event'
    PERSON_OF_INTEREST = 'person_of_interest'


def save_data(path_to_data_folder: str, sql_query: str, credentials: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_connection = get_db_connection(credentials)

    df = pd.read_sql(sql_query, db_connection)
    df_save_and_exit = get_save_and_exit_df(df)
    event_data = get_event_training_data(df_save_and_exit)
    df_with_means = get_means_df(df_save_and_exit)
    person_of_interest_data = get_person_of_interest_data(df_with_means)

    event_data_path = f'{path_to_data_folder}/event_training_data.csv'
    person_of_interest_data_path = f'{path_to_data_folder}/person_of_interest_training_data.csv'
    event_data.to_csv(event_data_path, index=False)
    person_of_interest_data.to_csv(person_of_interest_data_path, index=False)

    return event_data, person_of_interest_data


def upload_model(model: Union[IForest, TreeExplainer], path_to_model_folder: str, model_name: str):
    path_to_model = os.path.join(path_to_model_folder, f"{model_name}.joblib")
    joblib.dump(model, path_to_model)

    with tarfile.open(f'{path_to_model_folder}/{model_name}.tar.gz', 'w:gz') as file:
        file.add(path_to_model)

    return os.path.join(upload_to_s3(path_to_model_folder), f'{model_name}.tar.gz')


def deploy_model(
    model_data: str,
    entry_point: str,
    source_dir: str = "endpoint_code",
    *,
    endpoint_name: str = None,
    role: str,
) -> None:
    model = SKLearnModel(
        role=role,
        model_data=model_data,
        framework_version="1.2-1",
        py_version="py3",
        source_dir=os.path.join(os.path.dirname(__file__), source_dir),
        entry_point=entry_point,
    )
    predictor = model.deploy(
        instance_type="ml.t2.medium",
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        serverless_inference_config=ServerlessInferenceConfig(max_concurrency=10)
    )


def deploy_classification_models(
    classification_type: Classification,
    data: pd.DataFrame,
    endpoint_prefix: str,
    model_dir: str,
    role: str,
    *,
    prediction_only: bool = False,
    estimator_only: bool = False
) -> None:
    prediction_model = IForest(n_estimators=1000)
    prediction_model.fit(data)
    model_estimator = TreeExplainer(prediction_model, event_data)

    name = classification_type.value
    model_s3_uri = upload_model(prediction_model, model_dir, f'{name}_model')
    estimator_s3_uri = upload_model(model_estimator, model_dir, f'{name}_estimator')
    prediction_endpoint_name = f'{endpoint_prefix}-{name}-prediction-endpoint'
    estimator_endpoint_name = f'{endpoint_prefix}-{name}-estimator-endpoint'

    if prediction_only:
        logger.info('Deploying prediction model only.')
        deploy_model(
            model_s3_uri,
            f'{name}_inference.py',
            endpoint_name=prediction_endpoint_name,
            role=role
        )
    elif estimator_only:
        logger.info('Deploying estimator model only.')
        deploy_model(
            estimator_s3_uri,
            f'{name}_estimator.py',
            endpoint_name=estimator_endpoint_name,
            role=role
        )
    else:
        deploy_model(
            model_s3_uri,
            f'{name}_inference.py',
            endpoint_name=prediction_endpoint_name,
            role=role
        )
        deploy_model(
            estimator_s3_uri,
            f'{name}_estimator.py',
            endpoint_name=estimator_endpoint_name,
            role=role
        )


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--sagemaker-role', default=ROLE, help='SageMaker Role to use.')
    parser.add_argument('--model-dir', default=MODEL_FOLDER, help='Directory where model data is stored.')
    parser.add_argument('--data-dir', default=DATA_FOLDER, help='Directory where training data is stored.')
    parser.add_argument('--endpoint-prefix', default=ENDPOINT_NAME_PREFIX, help='Model endpoint names prefix.')
    parser.add_argument('--rds-secret-name', default=RDS_SECRET_NAME, help='Secrets Manager secret name that holds RDS credentials.')
    parser.add_argument('--use-env', action='store_true', help='Use credentials stored in env variables.')
    parser.add_argument('--use-local-data', action='store_true', help='Use training data pre loaded from RDS in data directory.')
    parser.add_argument('--prediction-only', action='store_true', help='Deploy prediction model only.')
    parser.add_argument('--estimator-only', action='store_true', help='Deploy estimator model only')
    parser.add_argument('--event-only', action='store_true', help='Deplpoy event level models only')
    parser.add_argument('--person_of_interest-only', action='store_true', help='Deploy person_of_interest level models only.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    if not args.use_env:
        secret = json.loads(get_secret(RDS_SECRET_NAME))
        credentials = Credentials(
            USERNAME=secret['username'],
            PASSWORD=secret['password'],
            HOST=secret['host'],
            DB=secret['dbname'],
            PORT=secret['port']
        )
    else:
        credentials = os.environ

    if not args.use_local_data:
        logger.info('Pulling data from RDS.')
        event_data, person_of_interest_data = save_data(args.data_dir, QUERY, credentials)
    else:
        logger.info('Using locally saved training data.')
        event_data = pd.read_csv(args.data_dir + '/event_training_data.csv')
        person_of_interest_data = pd.read_csv(args.data_dir + '/person_of_interest_training_data.csv')

    deploy_func = partial(
        deploy_classification_models,
        endpoint_prefix=args.endpoint_prefix,
        model_dir=args.model_dir,
        role=args.sagemaker_role,
        prediction_only=args.prediction_only,
        estimator_only=args.estimator_only
    )

    if args.event_only:
        deploy_func(Classification.EVENT, event_data)
    elif args.person_of_interest_only:
        deploy_func(Classification.PERSON_OF_INTEREST, person_of_interest_data)
    else:
        deploy_func(Classification.EVENT, event_data)
        deploy_func(Classification.PERSON_OF_INTEREST, person_of_interest_data)
