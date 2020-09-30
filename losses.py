from __future__ import print_function, division
import torch.nn.functional as F
import torch


def IoULoss(inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)  
    return 1 - IoU


def ComboLoss( inputs, targets, smooth=1, alpha= 1.5, CE_RATIO = 0.2):
    inputs=inputs.sigmoid()
    inputs=inputs.argmax(dim=1)
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    eps = 1e-3
    e=eps
    #True Positives, False Positives & False Negatives
    intersection = (inputs * targets).sum()    
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    inputs = torch.clamp(inputs, e, 1.0 - e)       
    out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
    weighted_ce = out.mean(-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
    return combo

    
def TverskyLoss(inputs, targets, smooth=1, alpha= 0.1, beta=0.9):
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.sigmoid(inputs)       
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    return 1 - Tversky


def FocalTverskyLoss(inputs, targets, smooth=1, alpha=0.1, beta=.9, gamma=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.sigmoid(inputs)       
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = (1 - Tversky)**gamma      
    return FocalTversky



def FocalLoss(inputs, targets, alpha=0.8, gamma=2, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.sigmoid(inputs)       
    inputs=inputs[:,1,...]
    #flatten label and prediction tensors
    inputs = inputs.contiguous().view(-1).float()
    targets = targets.contiguous().view(-1).float()
    
    #first compute binary cross-entropy 
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE    
    return focal_loss




def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds