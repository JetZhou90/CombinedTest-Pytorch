from models import DetectModel
import torch
import yaml, os
from experiment import EfficientExperiment
from pytorch_lightning import Trainer

with open('configs/detect.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


model = DetectModel(num_classes=config['model_params']['num_classes'], 
                        compound_coef=config['model_params']['compound_coef'], 
                        obj_list=config['model_params']['obj_list'], 
                        ratios=eval(config['model_params']['anchors_ratios']), 
                        scales=eval(config['model_params']['anchors_scales']))
experiment = EfficientExperiment(model, config['exp_params']) 
checkpoint = torch.load("logs/EfficientNet/version_2/checkpoints/_ckpt_epoch_188.ckpt",map_location='cpu')
experiment.load_state_dict(checkpoint['state_dict'])
model = experiment.model
model.requires_grad_(False)
model.eval()

for image in os.listdir('D:/Projects/seal_validation/result/'):
    path = 'D:/Projects/seal_validation/result/'+image
    save_path = './test/'+image
    model.detect_image(path,save_path,line_thickness=2,threshold = 0.7)