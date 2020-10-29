
import cv2
import numpy as np

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