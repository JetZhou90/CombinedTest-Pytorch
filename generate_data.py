
import cv2
import numpy as np
import random,json

def combine_seal(seals_list,label_list,back,roi_list,num_seals,save_name):
    save_file = {"filename":save_name, 'locations':[],'labels':[]}
    
    for num in range(num_seals):
        annotation = np.zeros((1, 5))
        index = random.randint(0,10)
        seal = seals_list[index]
        label = label_list[index]
        h,w = seal.shape[:2]
        if h<100 or w<100:
            seal = cv2.resize(seal,(int(w*1.5),int(h*1.5)))
        h,w = seal.shape[:2]
        roi_index = num % len(roi_list)
        xmin,ymin,xmax,ymax = roi_list[roi_index]
        if xmin>=xmax-w or ymin>=ymax-h:
            print('Fail')
            continue
        l_xmin = xmin + random.randint(0,5)
        l_ymin = ymin + random.randint(0,5)
        l_xmax = l_xmin + w
        l_ymax = l_ymin + h
        try:
            dst = process_combine(seal,back,l_xmin,l_ymin,l_xmax,l_ymax)
            save_file['locations'].append([l_xmin,l_ymin,l_xmax,l_ymax])   
            save_file['labels'].append(label)  
        except:
            print('Faile in num '+ str(num))
            continue
        back[l_ymin:l_ymax,l_xmin:l_xmax]=dst
        new_xmin = xmin if l_xmin - xmin > w else l_xmax
        new_xmax = l_xmin if l_xmin - xmin> w else xmax
        new_ymin = ymin if l_ymin - ymin > h else l_ymax
        new_ymax = l_ymin if l_ymin - ymin> h else ymax
        roi_list[roi_index] = [new_xmin,new_ymin,new_xmax,new_ymax]
    cv2.imwrite(save_name+'.png',back)
    with open(save_name+'.json','w') as f:
        json.dump(save_file, f)
         
def process_combine(img,back,xmin,ymin,xmax,ymax):
    h,w = img.shape[:2]
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,200,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    new_area = back[ymin:ymax,xmin:xmax]
    img_bg = cv2.bitwise_and(new_area,new_area,mask=mask)
    mask_img = cv2.bitwise_and(img,img,mask = mask_inv)
    dst = cv2.add(img_bg,mask_img)
    return dst