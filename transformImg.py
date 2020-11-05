import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from scipy.signal import convolve2d
 
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

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

def transfrom_img(seal_condition,resized_seal_img,min_mathch_count=10, flann_index_kdtree=1,trees=5,checks=50, thresh=0.35):
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
        if m.distance < thresh * n.distance:
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

def padding(image, target_size):
    if image.shape[-1]==3:
        h, w, c = image.shape
    else:
        h, w = image.shape[0],image.shape[1]
        c =1
    t_h, t_w = target_size
    if min(t_h, t_w) < max(h, w):
        print('Please change target size, it must be larger than both w and h')
        return image
    h_val = (t_h - h ) // 2
    w_val = (t_w - w ) // 2
    new_img = cv2.copyMakeBorder(image, h_val,t_h - h-h_val,w_val,t_w - w-w_val,cv2.BORDER_REPLICATE)
    return new_img

class PHash(object):
    @staticmethod
    def pHash(image):
        """
        get image pHash value
        """
        # 加载并调整图片为32x32灰度图片
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
 
        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img  # 填充数据
 
        # 二维Dct变换
        vis1 = cv2.dct(cv2.dct(vis0))
        # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
        vis1.resize((32, 32), refcheck=False)
 
        # 把二维list变成一维list
        img_list = list(chain.from_iterable(vis1))
 
        # 计算均值
        avg = sum(img_list) * 1. / len(img_list)
        avg_list = ['0' if i < avg else '1' for i in img_list]
    
        # 得到哈希值
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32*32, 4)])
 
    @staticmethod
    def hammingDist(s1, s2):
        """
        计算两张图片的汉明距离
        """
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        # raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    return np.mean(np.mean(ssim_map))

if __name__ == "__main__":
    img_path = 'data/unet/train/new_58-16_2850.png'
    mask_path = 'data/unet/train/new_58-16_2850.jpg'
    condition_path = '../cmb_seal/square_6.png'
    seal_condition = Image.open(condition_path)
    seal_condition = np.array(seal_condition)
    seal_condition = seal_condition[...,:3]
    h,w,c = seal_condition.shape
    val = max(h,w)
    seal_img = masked_image(img_path, mask_path)
    h,w,c = seal_img.shape
    val = max(val,h,w)
    resized_seal_img = padding(seal_img,(val,val))
    seal_condition = padding(seal_condition,(val,val))
    # resized_seal_img = cv2.resize(seal_image,(w,h))
    trans_img = transfrom_img(seal_condition,resized_seal_img,min_mathch_count=3)
    # resized_seal_img = cv2.resize(trans_img,(w,h))
    if trans_img is None:
        image_show(np.hstack([resized_seal_img,seal_condition]))
        print('Not Match')
    else:
        image_show(np.hstack([resized_seal_img,trans_img,seal_condition]))
        print('ssim:s',ssim(trans_img,seal_condition))
        hash1 = PHash.pHash(trans_img)
        hash2 = PHash.pHash(seal_condition)
        distance = PHash.hammingDist(hash1, hash2)
        print(distance)
        out_score = 1 - distance * 1. / (32 * 32 / 4)
        print('pHash Score:',out_score)
        print('Corrcoef:', np.corrcoef(trans_img.flatten(),seal_condition.flatten())[0][1])