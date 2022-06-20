import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random

class random_augmentation(object):
    def __init__(self):
        self.random_flip = A.OneOf(
            [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0)
            ],p=1.0)
        self.random_ShiftScaleRotate = A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.15, rotate_limit=60, p=1.0, border_mode=0
        )

    def __call__(self, image, gt):
        if random.random() >= 0.5:
            transformed = self.random_flip(image=image, mask=gt)
            image, gt = transformed['image'], transformed['mask']
        elif random.random() < 0.5:
            transformed = self.random_ShiftScaleRotate(image=image, mask=gt)
            image, gt = transformed['image'], transformed['mask']

        return image, gt

class Qia_Dataset(Dataset):
    """
    dataloader for QiaZong segmentation tasks
    """
    def __init__(self, image_root, gt_root, augmentation=None):
        self.augmentation = augmentation
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        # self.images = images
        # self.gts = gts
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        gt = gt / 255.0

        if self.augmentation != None:
            image, gt = self.augmentation(image, gt)
        image = self.transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batch_size, shuffle=True, num_workers=4, pin_memory=True, augmentation=None):
    # if split == 'train':
    #     dataset = Qia_Dataset(image_root, gt_root, augmentation)
    # elif split != 'train':
    #     dataset = Qia_Dataset(image_root, gt_root)
    dataset = Qia_Dataset(image_root, gt_root, augmentation)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    dataset_size = dataset.size
    return data_loader, dataset_size


def get_k_fold_data(k, i, image_root, label_root):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    images = np.load(image_root)
    labels = np.load(label_root)
    assert k > 1
    fold_size = images.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    image_train, label_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        image_part, label_part = images[idx, :], labels[idx]
        if j == i:  ###第i折作valid
            image_valid, label_valid = image_part, label_part
        elif image_train is None:
            image_train, label_train = image_part, label_part
        else:
            image_train = np.concatenate((image_train, image_part), axis=0)
            label_train = np.concatenate((label_train, label_part), axis=0)
            # image_train = torch.cat((image_train, image_part), dim=0)  # dim=0增加行数，竖着连接
            # label_train = torch.cat((label_train, label_part), dim=0)
    # print(X_train.size(),X_valid.size())
    train_dataset = Qia_Dataset(image_train, label_train)
    valid_dataset = Qia_Dataset(image_valid, label_valid)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True)
    # train_size = train_dataset.size
    # valid_size = valid_dataset.size
    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    path = '/home/songmeng/data/CXR_data/'
    image_root = path + 'data_train.npy'
    label_root = path + 'data_label.npy'
    images = np.load(image_root)
    img = images[0]
    print(img.shape)
    plt.imshow(img)
    plt.savefig('/home/songmeng/data/CXR_data/1.png')
    # for i in range(5):
    #     train_dataloader, train_size, valid_dataloader, valid_size = get_k_fold_data(5, i, image_root, label_root)
    #     print('{} fold'.format(i))
    #     print(len(train_dataloader), train_size)
    #     print(len(valid_dataloader), valid_size)
    # augmentation = random_augmentation()

    # data_loader, size = get_loader(image_root, gt_root, batch_size=4)
    # print(len(data_loader), size)

    # tt = Qia_Dataset(path + 'data_train.npy', path + 'mask_train.npy')
    #
    # for i in range(50):
    #     img, gt = tt.__getitem__(i)
    #
    #     img = torch.transpose(img, 0, 1)
    #     img = torch.transpose(img, 1, 2)
    #     img = img.numpy()
    #     gt = gt.numpy()
    #
    #     plt.imshow(img)
    #     plt.savefig('vis/' + str(i) + ".jpg")
    #
    #     plt.imshow(gt[0])
    #     plt.savefig('vis/' + str(i) + '_gt.jpg')
