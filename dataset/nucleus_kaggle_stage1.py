import os
import numpy as np

import torch
from PIL import Image
from torch.utils import data

#change this line to your own data root path
root = '/home/bingzhe/dataset/kaggel/stage1_train/'
def get_file_path(data_dir):
    data_path = os.listdir(data_dir)[0]
    data_path = os.path.join(data_dir, data_path)
    return data_path
def make_data_path_list(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_img_ids = os.listdir(root)
        for img_id in train_img_ids:
            img_path = os.path.join(root, img_id,'images')
            img_file_path = get_file_path(img_path)
            mask_path = os.path.join(root, img_id, 'mask_')
            mask_file_path = get_file_path(mask_path)
            item = (img_file_path, mask_file_path)
            items.append(item)
    return items

class nucleus_stage1(data.Dataset):
    def __init__(self, mode = 'train', joint_transform = None, transform = None, target_transform = None):
        super(nucleus_stage1, self).__init__()
        self.imgs = make_data_path_list(mode)
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask
    def __len__(self):
        return len(self.imgs)
def test_nuckeus_stage1():
    import extend_transforms
    import joint_transforms
    import torchvision.transforms as standard_transforms
    mean_std = mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale_((256, 256)),
        joint_transforms.RandomCrop(256),
        joint_transforms.RandomHorizontallyFlip()
    ])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extend_transforms.MaskToTensor()
    dataset = nucleus_stage1(joint_transform = train_joint_transform,
    transform = input_transform, target_transform = target_transform)
    img, mask = dataset[0]
    print(img.size)
    print(mask.size)
def test_make_data_path_list():
    items = make_data_path_list('train')
    for item in items:
        print(os.path.exists(item[0]) and os.path.exists(item[1]))
        print(item[1])

if __name__ == '__main__':
    test_nuckeus_stage1()

            