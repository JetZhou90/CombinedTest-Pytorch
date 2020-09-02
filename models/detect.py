import torch , cv2
from torch import nn
import numpy as np
from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors, BBoxTransform, ClipBoxes
from efficientdet.loss import FocalLoss
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes, compound_coef, obj_list, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.obj_list = obj_list
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
        num_anchors = len(self.aspect_ratios) * self.num_scales
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])
        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]
        _, p3, p4, p5 = self.backbone_net(inputs)
        features = (p3, p4, p5)
        features = self.bifpn(features)
        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)
        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
    
    def loss_function(self, classification, regression, anchors, annotations, **kwargs) -> dict:
        criterion = FocalLoss()
        cls_loss, reg_loss =criterion(classification, regression, anchors, annotations)
        loss = cls_loss + reg_loss
        return {'loss':loss, 'cls_loss':cls_loss, 'reg_loss':reg_loss}

    def detect_image(self, image_path, save_path, line_thickness=None, use_cuda=False, use_float16=False,threshold = 0.2, iou_threshold = 0.2):
        # replace this part with your project's anchor config
        max_size = self.input_sizes[self.compound_coef]
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
       
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=max_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        features, regression, classification, anchors = self.forward(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
        out = invert_affine(framed_metas, out)
        self.__save_image(out, ori_imgs, save_path, line_thickness=line_thickness,imwrite=True)

    def __save_image(self, preds, imgs, save_path, line_thickness=None, imwrite=True):
        color_list = standard_to_bgr(STANDARD_COLORS)
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
            imgs[i] = imgs[i].copy()
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score, line_thickness=line_thickness,color=color_list[get_index_label(obj, self.obj_list)])
            if imwrite:
                cv2.imwrite(save_path, imgs[i][..., ::-1])