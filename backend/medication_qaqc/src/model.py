from pydantic import BaseModel

class PredictionPayload_IPEX(BaseModel):
    trained_model_path: str = None
    data_folder: str = None
    batch_size: int = 5

class TrainPayload_IPEX(BaseModel):
    data_folder: str = None
    neg_class: int
    modeldir: str = None
    learning_rate: float
    epochs: int
    data_aug: int
