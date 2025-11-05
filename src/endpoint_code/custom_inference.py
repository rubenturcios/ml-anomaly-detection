import os
import joblib


def predict_fn(input_object, model) -> tuple:
    print("calling model")
    predictions = model.decision_function(input_object)
    return {
        'predicted_decision_scores': predictions,
        'fitted_decision_scores': model.decision_scores_
    }


def model_fn(model_dir):
    model_dir += '/model'
    print(os.listdir(model_dir))
    print("loading person_of_interest_model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "person_of_interest_model.joblib"))
    return loaded_model
