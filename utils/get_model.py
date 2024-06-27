import sys
sys.path.insert(0, '/models')
from models.unet import *
from models.resunet import *
from models.swinunet.vision_transformer import SwinUnet

def get_model(model_dict):
    # get model from dict with string
    model_type = globals()[model_dict["type"]]
    model_params = model_dict["params"]
    model = model_type(**model_params)
    return model