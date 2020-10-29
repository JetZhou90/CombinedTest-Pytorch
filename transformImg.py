import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def image_show(img, axis='off',title=None,save=None):
    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.title(title)
    plt.axis(axis) # 关掉坐标轴为 off
    if save is not None:
        matplotlib.use('Agg')
        plt.savefig(save)
    else:
        plt.show()

def masked_image(img_path, mask_path):
    image = Image.open(img_path)
    mask = Image.open(mask_path)
    img_data = np.asarray(image).astype(np.uint8)
    mask_data = np.asanyarray(mask).astype(np.uint8)
    mask_data[mask_data>0]=1
    masked_image = img_data * np.expand_dims(mask_data,axis=-1)
    masked_image[masked_image==0]=255
    return masked_image

def capture_pix_area(image, kernel_size=3,kernel_size2=11,area_thresh=.1, bias=0,use_blur=False):
    rect_list = []
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    cv2.imwrite('binary.jpg',thresh)
    (cnts,_) = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])
    for c in cnts:
        if cv2.contourArea(c)<avgCntArea*area_thresh:
            continue      
        (x,y,w,h) = cv2.boundingRect(c)
        xmin,xmax,ymin,ymax = x,x+w,y,y+h
        box = [xmin-bias,ymin-bias,xmax+bias,ymax+bias]
        rect_list.append(box)
    return image, rect_list

def Transfrom_img(seal_condition,resized_seal_img,min_mathch_count=10, flann_index_kdtree=1,trees=5,checks=50):
    sift = cv2.SIFT_create()
    gray_condition= cv2.cvtColor(seal_condition,cv2.COLOR_BGR2GRAY)
    gray_seal = cv2.cvtColor(resized_seal_img,cv2.COLOR_BGR2GRAY)
    psd_kp1, psd_des1 = sift.detectAndCompute(gray_condition, None)
    psd_kp2, psd_des2 = sift.detectAndCompute(gray_seal, None)
    index_params = dict(algorithm=flann_index_kdtree, trees=trees)
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.35*n.distance:
            goodMatch.append(m)
    if len(goodMatch)> min_mathch_count:
        # 获取关键点的坐标
        src_pts = np.float32([ psd_kp1[m.queryIdx].pt for m in goodMatch ]).reshape(-1,1,2)
        dst_pts = np.float32([ psd_kp2[m.trainIdx].pt for m in goodMatch ]).reshape(-1,1,2)
        h,w = gray_seal.shape
        transform_matrix = cv2.estimateAffine2D(dst_pts,src_pts, True)[0]
        transformed_candidate = cv2.warpAffine(resized_seal_img, transform_matrix,(w, h), borderValue=(255, 255, 255))
    else:
        print( "Not enough matches are found - %d/%d" % (len(goodMatch), min_mathch_count))
        return None
    return transformed_candidate

if __name__ == "__main__":
    img_path = 'data/unet/train/images/new_10-11_05.png'
    mask_path = 'data/unet/train/annotations/new_10-11_05.jpg'
    condition_path = '../cmb_seal/square_2.png'
    seal_condition = Image.open(condition_path)
    seal_condition = np.array(seal_condition)
    seal_condition = seal_condition[...,:3]
    h,w,c = seal_condition.shape
    seal_img = masked_image(img_path, mask_path)
    rect_list = capture_pix_area(seal_img,use_blur=True)[1]
    for box in rect_list:
        xmin,ymin,xmax,ymax = box
        seal_image = seal_img[ymin:ymax,xmin:xmax]
    resized_seal_img = cv2.resize(seal_image,(w,h))
    trans_img = Transfrom_img(seal_condition,resized_seal_img)
    if trans_img is None:
        print('Not Match')
    else:
        image_show(np.hstack([resized_seal_img,trans_img,seal_condition]))
        print(np.corrcoef(trans_img.flatten(),seal_condition.flatten())[0][1])