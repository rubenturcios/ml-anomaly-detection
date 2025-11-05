from datetime import datetime
import logging
import os
from typing import List, TypedDict
import traceback

import boto3
from botocore.exceptions import ClientError
from haversine import haversine, Unit
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.engine import Connection
import pandas as pd
import sagemaker


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))


class Credentials(TypedDict):
    USERNAME: str
    PASSWORD: str
    HOST: str
    DB: str
    PORT: str


def get_secret(secret_name: str) -> dict:
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager')

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']
    return secret


def get_db_connection(credentials: Credentials) -> Connection:
    driver_name = 'postgresql+psycopg2'
    url = URL.create(
        drivername=driver_name,
        username=credentials['USERNAME'],
        password=credentials['PASSWORD'],
        host=credentials['HOST'],
        port=credentials['PORT'],
        database=credentials['DB']
    )
    engine = create_engine(url)
    db_connection = engine.connect()
    return db_connection


def get_datetime(date_time: str) -> datetime:
    index = date_time.rfind(' ')
    date_time = date_time[:index] + date_time[index + 1:]
    return datetime.fromisoformat(date_time)


def get_elapsed_time(t1: str, t2: str) -> int:
    datetime_1 = get_datetime(t1)
    datetime_2 = get_datetime(t2)
    return (datetime_1 - datetime_2).seconds


def create_event_data_features(
    df_per_facility_uuid: List[pd.DataFrame]
) -> pd.DataFrame:
    processed_dfs = []

    for df in df_per_facility_uuid:
        try:
            processed = _process_facility_df(df)
        except Exception:
            print(f'Issue with data from: {df["facility_uuid"].unique()[0]}')
        else:
            processed_dfs.append(processed)

    df = pd.concat(processed_dfs, ignore_index=True)
    return df


def _process_facility_df(df_: pd.DataFrame) -> pd.DataFrame:
    number_of_save_and_exit_counts = []
    elapsed_times = []
    distances = []

    count = 0
    start_time = str()
    number_of_save_and_exit_count = 0

    for row in df_.sort_values(by='timestamp').index:
        count += 1

        if df_['event'][row] == 13:
            start_time = df_['timestamp'][row]
            elapsed_times.append(0)
            distances.append(0)
            number_of_save_and_exit_counts.append(0)
        elif df_['event'][row] == 9:
            number_of_save_and_exit_count += 1
            number_of_save_and_exit_counts.append(number_of_save_and_exit_count)

            end_time = df_['timestamp'][row]
            elapsed_times.append((end_time - start_time).seconds)

            event_lat = df_['event_latitude'][row]
            event_long = df_['event_longitude'][row]
            event = (event_lat, event_long)

            fac_lat = df_['facility_latitude'][row]
            fac_long = df_['facility_longitude'][row]
            facility = (fac_lat, fac_long)

            distances.append(haversine(event, facility, Unit.METERS))

    elapsed_times.reverse()
    distances.reverse()
    number_of_save_and_exit_counts.reverse()

    df_['elapsed_time'] = elapsed_times
    df_['distance'] = distances
    df_['save_and_exit_count'] = number_of_save_and_exit_counts
    return df_


def get_save_and_exit_df(df: pd.DataFrame) -> pd.DataFrame:
    df_per_facility_uuid = [y for _, y in df.groupby('facility_uuid')]
    df = create_event_data_features(df_per_facility_uuid)
    df_save_and_exit_only = df[df['event'] == 9]
    return df_save_and_exit_only.reset_index(drop=True)


def get_event_training_data(df_save_and_exit_only: pd.DataFrame) -> pd.DataFrame:
    data = df_save_and_exit_only[['elapsed_time', 'distance', 'revision', 'save_and_exit_count']]
    return data


def get_means_df(df_save_and_exit_only: pd.DataFrame) -> pd.DataFrame:
    dfs_grouped_by_person_of_interest = [y for _, y in df_save_and_exit_only.groupby('person_of_interest_id')]

    for df_ in dfs_grouped_by_person_of_interest:
        df_.sort_values(by='timestamp', ascending=False, inplace=True)
        window = df_.rolling(window='7d', on='timestamp', closed='both')
        mean_distance = window.distance.mean()
        mean_elapsed_time = window.elapsed_time.mean()
        mean_revision = window.revision.mean()
        mean_save_and_exit_count = window.save_and_exit_count.mean()

        df_['mean_distance'] = mean_distance.values
        df_['mean_elapsed_time'] = mean_elapsed_time.values
        df_['mean_revision'] = mean_revision.values
        df_['mean_save_and_exit_count'] = mean_save_and_exit_count.values

    df_with_means = pd.concat(dfs_grouped_by_person_of_interest)
    return df_with_means


def get_prediction_means_df(df_save_and_exit_only: pd.DataFrame) -> pd.DataFrame:
    person_of_interest_rows = []
    dfs_grouped_by_person_of_interest = [y for _, y in df_save_and_exit_only.groupby('person_of_interest_id')]

    for df_ in dfs_grouped_by_person_of_interest:
        df_.sort_values(by='timestamp', ascending=False, inplace=True)
        person_of_interest_id = df_['person_of_interest_id'].unique()[0]
        mean_distance = df_.distance.mean()
        mean_elapsed_time = df_.elapsed_time.mean()
        mean_revision = df_.revision.mean()
        mean_save_and_exit_count = df_.save_and_exit_count.mean()

        person_of_interest_rows.append(
            pd.Series(
                {
                    'person_of_interest_id': person_of_interest_id,
                    'mean_elapsed_time': mean_elapsed_time,
                    'mean_distance': mean_distance,
                    'mean_revision': mean_revision,
                    'mean_save_and_exit_count': mean_save_and_exit_count
                }
            )
        )

    return pd.DataFrame(person_of_interest_rows)


def get_person_of_interest_data(df_with_means: pd.DataFrame) -> pd.DataFrame:
    data = df_with_means[
        [
            'mean_distance',
            'mean_elapsed_time',
            'mean_revision',
            'mean_save_and_exit_count'
        ]
    ]
    return data


def upload_to_s3(folder_path: str, prefix: str = 'Anomaly-Detection') -> str:
    folder_path = folder_path.lstrip('./')
    sagemaker_session = sagemaker.Session()
    train_input = sagemaker_session.upload_data(
        folder_path,
        key_prefix="{}/{}".format(prefix, folder_path)
    )
    return train_input
