import torch
import torch.nn as nn
import argparse
import os
import imageio
import torch.nn.functional as F
from thop import profile

from utils.validation import *
from dataset import get_loader
from network.own_net import Net
from train import get_args_parser

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test(model, test_dataloader, prediction_path):
    dice_bank = []
    iou_bank = []

    for i, sample in enumerate(test_dataloader, start=1):
        image, gt = sample
        image = image.cuda()

        _, _, pred = model(image)
        pred = F.sigmoid(pred).data.cpu().numpy().squeeze()
        gt = gt.data.cpu().numpy().squeeze()
        gt = 1 * (gt > 0.5)
        pred = 1 * (pred > 0.5)

        IoU = mean_iou(pred, gt)
        dice = dice_coefficient(pred, gt)
        dice_bank.append(dice)
        iou_bank.append(IoU)

        image = image.data.cpu().numpy().squeeze().transpose(1,2,0)

        imageio.imwrite(prediction_path+str(i).rjust(2,'0')+'_pred.png', pred)
        imageio.imwrite(prediction_path+str(i).rjust(2,'0')+'_gt.png', gt)
        imageio.imwrite(prediction_path + str(i).rjust(2, '0') + '_image.png', image)
        print('image {}: dice: {:.4f}, IoU: {:.4f}'.format(i, dice, IoU))

    Dice = np.mean(dice_bank)
    mIoU = np.mean(iou_bank)
    print('Dice: {:.4f}, mIoU: {:.4f}'.format(Dice, mIoU))

    return Dice, mIoU

if __name__ == '__main__':
    parser = get_args_parser()
    cfg = parser.parse_args()

    test_image = cfg.root + 'data_test.npy'
    test_gt = cfg.root + 'mask_test.npy'
    test_dataloader, test_size = get_loader(image_root=test_image,
                                            gt_root=test_gt,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=False,
                                            augmentation=None)
    print('[Test image size: {}]'.format(test_size))

    # model_list = os.listdir(cfg.model_path)
    # for i in model_list:
    #     if i[]

    device = torch.device('cuda:0')
    model = Net(num_classes=1).cuda()
    model.to(device)
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    input = torch.randn(4, 3, 512, 512).cuda()
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)

    # cfg.benchmark = 'qiazong_data'
    # cfg.model_name = 'unet++'
    # prediction_path = cfg.save_path + '/' + cfg.benchmark + '/' + cfg.model_name + '/'
    # if not os.path.exists(prediction_path):
    #     os.makedirs(prediction_path)
    #
    # test(model, test_dataloader, prediction_path)