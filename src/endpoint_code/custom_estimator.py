import os
import joblib
import pandas as pd


def predict_fn(input_object, model) -> tuple:
    print("calling model")
    return pd.DataFrame(
        model.shap_values(input_object) * -1,
        columns=[
            'model_impact_score_elapsed_time',
            'model_impact_score_distance',
            'model_impact_score_revision',
            'model_impact_score_save_and_exit_count',
        ]
    ).to_dict()


def model_fn(model_dir):
    model_dir += '/model'
    print(os.listdir(model_dir))
    print("loading person_of_interest_estimator.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "person_of_interest_estimator.joblib"))
    return loaded_model
