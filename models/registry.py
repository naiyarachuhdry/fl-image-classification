from .base import BaseModel
from .mobilenet import MobileNetModel
from .resnet import ResNetModel
from .cnn import CNNModel
from .vgg import VGGModel
from .resnet18_mlb import ResNet18MLBModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "mobilenet": MobileNetModel,
    "resnet": ResNetModel,
    "cnn": CNNModel,
    "vgg": VGGModel, 
    "resnet18_mlb": ResNet18MLBModel,
}