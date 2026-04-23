import json
import logging
import os

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def init():
    global model, inputs_collector, outputs_collector
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.sklearn.load_model(model_path)
    logger.info("Model loaded from %s", model_path)

    try:
        from azureml.ai.monitoring import Collector
        inputs_collector = Collector(name="model_inputs")
        outputs_collector = Collector(name="model_outputs")
    except ImportError:
        logger.warning("azureml.ai.monitoring not available, data collection disabled")
        inputs_collector = None
        outputs_collector = None


def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = data.get("input_data", data)

        if isinstance(input_data, dict) and "data" in input_data:
            input_data = input_data["data"]

        input_array = np.array(input_data)

        # Log inputs
        columns = ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
                    "TricepsThickness", "SerumInsulin", "BMI",
                    "DiabetesPedigree", "Age"]
        input_df = pd.DataFrame(input_array, columns=columns)
        if inputs_collector:
            inputs_collector.collect(input_df)

        predictions = model.predict(input_array)

        # Log outputs
        output_df = pd.DataFrame({"prediction": predictions.tolist()})
        if outputs_collector:
            outputs_collector.collect(output_df)

        return json.dumps({"result": predictions.tolist()})
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return json.dumps({"error": str(e)})
