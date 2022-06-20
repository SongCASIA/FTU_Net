import numpy as np
import cv2
import os
import albumentations as A
import random

root = '/home/songmeng/data/ISIC2017/' # change to your data folder path
data_f = ['train/', 'test/']
mask_f = ['train_gt/', 'test_gt/']
# set_size = [2000, 150, 600]
set_size = [2000, 600]
save_name = ['train', 'test']
# save_path = '/home/songmeng/SwinUnet/data/'
val_list = []
test_list = []

height = 512
width = 512

# class random_augmentation(object):
#     def __init__(self):
#         self.random_flip = A.OneOf(
#             [
#                 A.HorizontalFlip(p=1.0),
#                 A.VerticalFlip(p=1.0)
#             ],p=1.0)
#         self.random_ShiftScaleRotate = A.ShiftScaleRotate(
#             shift_limit=0.2, scale_limit=0.15, rotate_limit=60, p=1.0, border_mode=0
#         )
#
#     def __call__(self, image, gt, r):
#         if r >= 0.5:
#             transformed = self.random_flip(image=image, mask=gt)
#             image, gt = transformed['image'], transformed['mask']
#         elif r < 0.5:
#             transformed = self.random_ShiftScaleRotate(image=image, mask=gt)
#             image, gt = transformed['image'], transformed['mask']
#
#         return image, gt
#
# def process():
# 	augmentation = random_augmentation()
# 	length = 500
# 	image_path = root + 'image/'
# 	label_path = root + 'label/'
# 	images = np.uint8(np.zeros([length, height, width, 3]))
# 	labels = np.uint8(np.zeros([length, height, width]))
# 	count = 0
#
# 	for i in os.listdir(image_path):
# 		img = cv2.imread(image_path + i)
# 		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 		img = cv2.resize(img, (width, height))
#
# 		label_p = label_path + i
# 		label = cv2.imread(label_p, 0)
# 		label = cv2.resize(label, (width, height))
#
# 		images[count] = img
# 		labels[count] = label
# 		count += 1
# 		print(count, i)
#
# 	for i in os.listdir(image_path):
# 		if count < length:
# 			img = cv2.imread(image_path + i)
# 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 			img = cv2.resize(img, (width, height))
# 			label_p = label_path + i
# 			label = cv2.imread(label_p, 0)
# 			label = cv2.resize(label, (width, height))
# 			r = random.random()
# 			img, label = augmentation(img, label, r)
# 			images[count] = img
# 			labels[count] = label
# 			count += 1
# 			print(count, i)
#
# 	np.save('{}/data_image.npy'.format(root), images)
# 	np.save('{}/data_label.npy'.format(root), labels)
#
# if __name__ == '__main__':
# 	process()


for j in range(2):

	print('processing ' + data_f[j] + '......')
	count = 0
	length = set_size[j]
	imgs = np.uint8(np.zeros([length, height, width, 3]))
	masks = np.uint8(np.zeros([length, height, width]))

	path = root + data_f[j]
	mask_p = root + mask_f[j]

	for i in os.listdir(path):
		img = cv2.imread(path+i)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (width, height))

		m_path = mask_p + i.replace('.jpg', '_segmentation.png')
		# m_path = mask_p + i
		mask = cv2.imread(m_path, 0)
		mask = cv2.resize(mask, (width, height))

		imgs[count] = img
		masks[count] = mask

		count +=1
		print(count, i)


	np.save('{}/data_{}.npy'.format(root, save_name[j]), imgs)
	np.save('{}/mask_{}.npy'.format(root, save_name[j]), masks)
