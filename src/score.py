import json
import logging
import os

import mlflow
import numpy as np

logger = logging.getLogger(__name__)


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.sklearn.load_model(model_path)
    logger.info("Model loaded from %s", model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = data.get("input_data", data)

        if isinstance(input_data, dict) and "data" in input_data:
            input_data = input_data["data"]

        predictions = model.predict(np.array(input_data))
        return json.dumps({"result": predictions.tolist()})
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return json.dumps({"error": str(e)})
