import os,sys,glob,shutil,json
import cv2
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

# 创建路径，
train_path = glob.glob('./*.png')
train_path.sort()
train_json =json.load(open('./train.json'))
train_label = [train_json[x]['label']for x in train_json]

#
class SvhnDdateset(Dataset):
    def __init__(self,img_path,img_label,transform = None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(train_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = np.array(self.img_label[index],dtype=np.int)
        label = list(label)+(6-len(label))*[10]
        label = torch.from_numpy(np.array(label[:6]))
        return img,label
    def __len__(self):
        return len(self.img_path)

train_loader = torch.utils.data.DataLoader(
    SvhnDdateset(train_path,train_label,
                 transform = transforms.Compose([
                     transforms.Resize((64,128)),
                     transforms.ColorJitter(0.3,0.2,0.2),
                     transforms.RandomRotation(5),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])),
    batch_size=10,
    shuffle=False,
    num_workers=10,
)

for data in train_loader:
    break