from typing import Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    output_data_path: str
    model_path: str
    model_type: str
    feature_params: FeatureParams
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = False
    mlflow_uri: str = "http://18.156.5.226/"
    mlflow_experiment: str = "inference_demo"


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
