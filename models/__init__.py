from .base import BaseVAE
from .ccvae import ConCVAE
from .cvae import ConditionalVAE
from .detect import EfficientDetBackbone
from .unet import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, Deeplabv3_Resnet101,Deeplabv3_Resnet50
from .jetnet import JetNet

DetectModel = EfficientDetBackbone
BaseUnet = U_Net

unet_models = {
    "U_Net" : U_Net,
    "NestedUNet":NestedUNet,
    "deeplabv3_resnet101":Deeplabv3_Resnet101,
    "deeplabv3_resnet50":Deeplabv3_Resnet50
    }

vae_models = {
    'ConditionalVAE':ConditionalVAE,
    'ConCVAE':ConCVAE,
    'jetNet':JetNet
}

