import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_classes', type=int, default=1, help='class number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--root', type=str, default='/home/songmeng/data/qiazong_data/', help='root path')
    parser.add_argument('--model_save', type=str, default='/home/songmeng/snapshot/',
                        help='path to save model')
    parser.add_argument('--benchmark', type=str, default='Kvasir', help='dataset name for training')
    parser.add_argument('--model_name', type=str, default='TransFuse', help='selected model for training')
    parser.add_argument('--is_pretrained', type=bool, default=False, help='use pretrained model or not')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    parser.add_argument('--save_path', type=str, default='/home/songmeng/prediction/',
                       help='path to save predicted image')
    parser.add_argument('--model_path', type=str,
                       default='/home/songmeng/snapshot/qiazong_data/own/epoch-136-best_dice-0.9281.pth')
    # parser.add_argument('--optimizer_name', type=str, default='adamw', help='choose optimizer')
    # parser.add_argument('--log_path', type=str, default='/home/songmeng/OwnNet-v3/run-log/',
    #                     help='path to save scalar')
    # parser.add_argument('--decay_epoch', type=int, default=20,
    #                     help='epoch interval to decay lr, used in StepLRScheduler')
    # parser.add_argument('--lr_scheduler_name', type=str, default='cosine',
    #                     help='lr scheduler name')

    return parser