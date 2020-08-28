from .base import BaseVAE
from .ccvae import ConCVAE
from .cvae import ConditionalVAE
from .detect import EfficientDetBackbone
from .unet import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net


DetectModel = EfficientDetBackbone
BaseUnet = U_Net

unet_models = {
    "U_Net" : U_Net,
    "NestedUNet":NestedUNet
    }

vae_models = {
    'ConditionalVAE':ConditionalVAE,
    'ConCVAE':ConCVAE
}

