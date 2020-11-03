import torch, os, random,cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import json

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]
            rows, cols, channels = image.shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample = {'img': image, 'annot': annots}
        return sample
        

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class Detect_Dataset_folder(Dataset):
    def __init__(self, img_dir, ann_dir, obj_list, input_size=512, transform=None):
        self.images = sorted(os.listdir(img_dir))
        self.ann    = sorted(os.listdir(ann_dir))
        self.images_dir = img_dir
        self.obj_list = obj_list
        self.ann_dir = ann_dir
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                Augmenter(),
                Resizer(input_size)
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        path = os.path.join(self.images_dir , self.images[image_index])
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        path = os.path.join(self.ann_dir , self.ann[image_index])
        annotations = np.zeros((0, 5))
        if path.split('.')[-1] == 'xml':
            tree = ET.parse(path)
            for elem in tree.iter():
                for attr in list(elem):
                    annotation = np.zeros((1, 5))
                    if 'name' in attr.tag:
                        annotation[0, 4] = 0 if attr.text =='icon' else 1
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                annotation[0,0] =  int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                annotation[0,1] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                annotation[0,2] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                annotation[0,3] = int(round(float(dim.text)))
                    annotations = np.append(annotations, annotation, axis=0)
        else:
            with open(path, 'rb') as f:
                jsonfile = json.load(f)
            label_list = jsonfile['labels']
            for i , (xmin,ymin,xmax,ymax) in enumerate(jsonfile['locations']):
                annotation = np.zeros((1, 5))
                annotation[0, 4] = self.obj_list.index(label_list[i])
                # print(annotation[0,:-1].shape)
                annotation[0,0] = xmin
                annotation[0,1] = ymin
                annotation[0,2] = xmax
                annotation[0,3] = ymax
                annotations = np.append(annotations, annotation, axis=0)
        return annotations


class Unet_Dataset_folder(Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        imH = Image Height (default:128) 
        imW = Image Width  (default:128) 
        transformI = Input Images transformation (default: None)
    Output:
        img    = Transformed images
        la_img = Transformed labels"""

    def __init__(self, images_dir, labels_dir, one_hot=True, num_class=4, imH=128, imW=128, transformI = None, transformL= None):
        self.images_dir = images_dir
        self.images = sorted(os.listdir(self.images_dir))
        self.labels_dir = labels_dir
        self.labels = sorted(os.listdir(self.labels_dir))
        self.transformI = transformI
        self.imH = imH
        self.imW = imW
        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = transforms.Compose([
                transforms.Resize((self.imH,self.imW)),
                # transforms.RandomRotation((-10,10)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

            ])
        if self.transformL:
            self.tl = self.transformL
        else:
            self.tl = transforms.Compose([
                transforms.Resize((self.imH,self.imW)),
                # transforms.RandomRotation((-10,10)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        
        self.one_hot = one_hot
        self.num_class = num_class
        
    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)   # reshape 为向量
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
        size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
        return ones.view(*size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        i1        = Image.open(os.path.join(self.images_dir , self.images[i])) 
        label_img = Image.open(os.path.join(self.labels_dir , self.labels[i]))
        seed=np.random.randint(0,100) # make a seed with numpy generator 
        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.tx(i1)
        # la_img = self.tl(label_img)
        if self.one_hot:
            tl = transforms.Compose([
                transforms.Resize((self.imH,self.imW))])
            resize_label_img = tl(label_img)
            label_img_data = np.array(resize_label_img)
            label_img_tensor = torch.LongTensor(label_img_data)
            label_img_one_hot = get_one_hot(label_img_tensor, self.num_class)
            la_img = label_img_one_hot.permute(2,1,0)
        else:
            la_img = self.tl(label_img)
        return img, la_img


class VAE_Dataset_folder(Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        imH = Image Height (default:128) 
        imW = Image Width  (default:128) 
        transformI = Input Images transformation (default: None)
    Output:
        img    = Transformed images
        la_img = Transformed labels"""

    def __init__(self, images_dir, labels_dir=None, imH=128, imW=128, transformI = None):
        self.images_dir = images_dir
        self.images = sorted(os.listdir(self.images_dir))
        if labels_dir is not None:
            self.labels_dir = labels_dir
        else:
            self.labels_dir = images_dir
        self.labels = sorted(os.listdir(self.labels_dir))
        self.transformI = transformI
        self.imH = imH
        self.imW = imW
        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = transforms.Compose([
                transforms.Resize((self.imH,self.imW)),
                transforms.RandomRotation((-10,10)),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
                # transforms.CenterCrop(96),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(os.path.join(self.images_dir , self.images[i])) 
        label_img = Image.open(os.path.join(self.labels_dir , self.labels[i]))
        if self.images[i][0:3]=='Aby':
            label = torch.Tensor([1,0])
        else:
            label = torch.Tensor([0,1])
        seed=np.random.randint(0,100) # make a seed with numpy generator 
        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.tx(i1)
        la_img = self.tx(label_img)
        return img, la_img


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    imgs = imgs.permute(0, 3, 1, 2)
    return {'img': imgs, 'annot': annot_padded, 'scale': scales}