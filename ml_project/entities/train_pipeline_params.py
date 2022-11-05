from typing import Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from .model_params import RFParams, LogregParams
from .transform_params import TransformParams
from marshmallow_dataclass import class_schema
from hydra.core.config_store import ConfigStore
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    train_x_path: str
    train_y_path: str
    test_x_path: str
    test_y_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    transform_params: Optional[TransformParams] = None
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = False
    mlflow_uri: str = "http://18.156.5.226/"
    mlflow_experiment: str = "inference_demo"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
    
def register_train_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=TrainingPipelineParams)
    cs.store(
        group="model",
        name="rf",
        node=RFParams
    )
    cs.store(
        group="model",
        name="logreg",
        node=LogregParams
    )