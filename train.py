import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils.loss_func import DiceLoss, FocalLoss, GHMC_Loss, structure_loss, Multi_DiceLoss
from dataset import get_loader, random_augmentation, get_k_fold_data
from utils.validation import mean_iou, dice_coefficient, rc_pre_F1_score
from network.own_net import Net
from config import get_args_parser
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = get_args_parser()
cfg = parser.parse_args()
writer = SummaryWriter('./logs/exp11')

def adjust_learning_rate(optimizer, epoch):
    """
    decayed by 10 every 30 epoch
    """
    lr = cfg.lr * (0.9 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print('[learning = {}]'.format(lr))

def validate(val_dataloader, model, criterion):
    loss_bank = []
    dice_bank = []
    iou_bank = []
    # hd95_bank = []
    recall_bank = []
    precision_bank = []
    F1_score_bank = []

    for i, sample in enumerate(val_dataloader, start=1):
        image, gt = sample
        image = image.cuda()
        gt = gt.cuda()

        with torch.no_grad():
            # pred = model(image)
            _, _, pred = model(image)

        loss = criterion(pred, gt)

        pred = F.sigmoid(pred).data.cpu().numpy().squeeze()
        gt = gt.data.cpu().numpy().squeeze()
        gt = 1 * (gt > 0.5)
        pred = 1 * (pred > 0.5)

        IoU = mean_iou(pred, gt)
        dice = dice_coefficient(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        recall, precision, F1_score = rc_pre_F1_score(pred, gt)

        loss_bank.append(loss.item())
        dice_bank.append(dice)
        iou_bank.append(IoU)
        # hd95_bank.append(hd95)
        recall_bank.append(recall)
        precision_bank.append(precision)
        F1_score_bank.append(F1_score)

    return np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(recall_bank), np.mean(precision_bank), np.mean(F1_score_bank)

def trainer(train_dataloader, val_dataloader, model, optimizer,
            criterion, epoch, save_path, best_iou, best_dice):
    adjust_learning_rate(optimizer, epoch=epoch+1)
    device = torch.device('cuda:0')
    model.to(device)
    model.train()

    loss_bank = []

    for i, sample in enumerate(train_dataloader, start=1):
        images, gts = sample
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # pred = model(images)

        # loss = criterion(pred, gts)

        lateral_map_1, lateral_map_2, lateral_map_3 = model(images)
        loss1 = criterion(lateral_map_1, gts)
        loss2 = criterion(lateral_map_2, gts)
        loss3 = criterion(lateral_map_3, gts)
        loss = 0.2 * loss1 + 0.3 * loss2 + 0.5 * loss3

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
        optimizer.step()
        # lr_scheduler.step_update(epoch * len(train_dataloader) + i)

        loss_bank.append(loss.item())
    mean_loss = np.mean(loss_bank)

    print('{} Epoch, Train Loss: {:.4f}'.format(epoch, mean_loss))

    model.eval()
    val_loss, dice, mIoU, recall, precision, F1_score = validate(val_dataloader, model, criterion)

    writer.add_scalar('dice', dice, epoch)
    writer.add_scalar('mIoU', mIoU, epoch)
    # writer.add_scalar('/hd95', hd95, epoch)
    writer.add_scalar('recall', recall, epoch)
    writer.add_scalar('precision', precision, epoch)
    writer.add_scalar('F1_score', F1_score, epoch)
    writer.add_scalar('loss', mean_loss, epoch)

    print('Val Loss: {:.4f}, Dice: {:.4f}, mIoU: {:.4f}, recall: {:.4f}, precision: {:.4f}, F1_score: {:.4f}'.format(
        val_loss, dice, mIoU, recall, precision, F1_score))

    # if mean_loss < best_loss:
    #     print('[new best loss: {:.4f}]'.format(mean_loss))
    #     best_loss = mean_loss
    #     save_model_path = os.path.join(save_path, 'epoch-'+str(epoch).rjust(3, '0')+'.pth')
    #     torch.save(model.state_dict(), save_model_path)
    #     print('[Save model to : {}]'.format(save_model_path))

    if dice > best_dice:
        best_dice = dice
        best_iou = mIoU
        print('[new best dice: {:.4f}, best iou: {:.4f}]'.format(best_dice, best_iou))
        save_model_path = os.path.join(save_path, 'epoch-'+str(epoch).rjust(3, '0')+'.pth')
        torch.save(model.state_dict(), save_model_path)
        print('[Save model to : {}]'.format(save_model_path))

    return best_dice, best_iou



if __name__ == '__main__':
    train_image = cfg.root + 'data_train.npy'
    train_gt = cfg.root + 'mask_train.npy'
    val_image = cfg.root + 'data_test.npy'
    val_gt = cfg.root + 'mask_test.npy'
    data_augmentation = random_augmentation()
    train_dataloader, train_size = get_loader(image_root=train_image,
                                              gt_root=train_gt,
                                              batch_size=cfg.batch_size,
                                              shuffle=True,
                                              pin_memory=False,
                                              augmentation=data_augmentation)
    val_dataloader, val_size = get_loader(image_root=val_image,
                                          gt_root=val_gt,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4,
                                          pin_memory=False,
                                          augmentation=None)
    print('[Train dataset size: {}]'.format(train_size))
    print('[Train iter every epoch: {}]'.format(len(train_dataloader)))
    print('[Val dataset size: {}]'.format(val_size))

    # images_path = cfg.root + 'data_image.npy'
    # labels_path = cfg.root + 'data_label.npy'

    model = Net(num_classes=1).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, betas=(0.5, 0.999))
    # optimizer = build_optimizer(config=cfg, model=model)
    # lr_scheduler = build_scheduler(config=cfg, optimizer=optimizer, n_iter_per_epoch=len(train_dataloader))

    # criterion = Multi_DiceLoss(n_classes=1)
    criterion = structure_loss()
    criterion = DiceLoss()

    epoch = cfg.epoch
    cfg.benchmark = 'qiazong_data'
    cfg.model_name = 'own'
    save_path = cfg.model_save + cfg.benchmark + '/' + cfg.model_name + '/' + '-cnn+trans-'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_loss = 1
    best_dice = 0.5
    best_iou = 0

    for i in range(epoch):
        best_dice, best_iou = trainer(train_dataloader, val_dataloader, model, optimizer,
                                criterion, epoch=i+1, save_path=save_path, best_iou=best_iou, best_dice=best_dice)
        print('-' * 60)
        # scheduler.step(best_loss)  # learning rate decay
    writer.close()

    print('Train End, best dice: {:.4f}, best mIoU: {:.4f}'.format(best_dice, best_iou))