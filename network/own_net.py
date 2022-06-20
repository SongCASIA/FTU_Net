from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
from collections import OrderedDict
from torch.nn.modules.utils import _pair
from functools import partial
from torchvision.models import resnet34, resnet50
from timm.models.layers import to_2tuple
from network.pyramid_vision_transformer import PyramidVisionTransformer
from network.swin_transformer import SwinTransformer, SwinTransformerBlock, PatchEmbed

# from inplace_abn import InPlaceABN, InPlaceABNSync

# class MLP(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_ch, mid_ch)
#         self.bn1 = nn.BatchNorm2d(mid_ch)
#         self.act1 = nn.GELU()
#         self.fc2 = nn.Linear(mid_ch, out_ch)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         self.act2 = nn.GELU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.act2(x)
#         return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        y = self.conv(x)
        return y

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Up_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up_Conv, self).__init__()
        self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.up(x)
        return y

class ResNet_encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet_encoder, self).__init__()
        # self.resnet = resnet34()
        self.resnet = resnet34(pretrained=pretrained)
        if pretrained:
            self.resnet.load_state_dict(torch.load('/home/songmeng/pretrained/resnet_model/resnet34-333f7ec4.pth'))
        self.resnet.fc = nn.Identity()
        # self.resnet.layer4 = nn.Identity()
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        features = []
        x_u = self.resnet.conv1(x)
        x_u = self.resnet.bn1(x_u)
        x_u_0 = self.resnet.relu(x_u)
        # features.append(x_u)
        x_u = self.resnet.maxpool(x_u_0)

        x_u_4 = self.resnet.layer1(x_u)
        x_u_4 = self.drop(x_u_4)
        features.append(x_u_4)
        # draw_features(8, 8, x_u_4.detach().cpu().numpy(), '/home/songmeng/heatmap/layer1.png')

        x_u_3 = self.resnet.layer2(x_u_4)
        x_u_3 = self.drop(x_u_3)
        features.append(x_u_3)
        # draw_features(8, 16, x_u_3.detach().cpu().numpy(), '/home/songmeng/heatmap/layer2.png')

        x_u_2 = self.resnet.layer3(x_u_3)
        x_u_2 = self.drop(x_u_2)
        features.append(x_u_2)
        # draw_features(16, 16, x_u_2.detach().cpu().numpy(), '/home/songmeng/heatmap/layer3.png')

        x_u_1 = self.resnet.layer4(x_u_2)
        x_u_1 = self.drop(x_u_1)
        features.append(x_u_1)
        # draw_features(16, 32, x_u_1.detach().cpu().numpy(), '/home/songmeng/heatmap/layer4.png')

        features.append(x_u_0)
        # draw_features(8, 8, x_u_0.detach().cpu().numpy(), '/home/songmeng/heatmap/layer0.png')

        # features = features[::-1]

        # x_u = self.resnet.layer3(x_u_1)
        # x_u = self.drop(x_u)  # 8, 1024, 32, 32

        return features

# class pvt_tiny_encoder(PyramidVisionTransformer):
#     def __init__(self, **kwargs):
#         super(pvt_tiny_encoder, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
#             sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
#
# class pvt_small_encoder(PyramidVisionTransformer):
#     def __init__(self, **kwargs):
#         super(pvt_small_encoder, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
#             sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
#
# class swin_tiny_encoder(SwinTransformer):
#     def __init__(self, **kwargs):
#         super(swin_tiny_encoder, self).__init__(
#             img_size=1024, patch_size=4, in_chans=3, num_classes=1,
#             embed_dim=64, depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16],
#             window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#             norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
#             use_checkpoint=False
#         )

class swin_small_encoder(SwinTransformer):
    def __init__(self, **kwargs):
        super(swin_small_encoder, self).__init__(
            img_size=512, patch_size=4, in_chans=3, num_classes=1,
            embed_dim=64, depths=[2, 2, 18, 2], num_heads=[2, 4, 8, 16],
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
            use_checkpoint=False
        )

# class FeatureSelectionModule(nn.Module):
#     def __init__(self, in_ch, r_ratio):
#         super(FeatureSelectionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(nn.Conv2d(in_ch, in_ch // r_ratio, kernel_size=1, bias=True),
#                                 nn.ReLU(inplace=True),
#                                 nn.Conv2d(in_ch // r_ratio, in_ch, kernel_size=1, bias=True))
#         self.sigmoid = nn.Sigmoid()
#         self.conv = nn.Conv2d(in_ch, in_ch // r_ratio, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#
#         att = avg_out + max_out
#         att = self.sigmoid(att)
#         feat = torch.mul(x, att)
#         x = x + feat
#         out = self.conv(x)
#         return out

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class Fusion_block(nn.Module):
    def __init__(self, in_dim):
        super(Fusion_block, self).__init__()
        # self.query_conv = FeatureSelectionModule(in_ch=in_dim, r_ratio=8)
        # self.key_conv = FeatureSelectionModule(in_ch=in_dim, r_ratio=8)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim*2, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(y)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        fuse = torch.cat([x, y], dim=1)
        # fuse = x + y
        proj_value = self.value_conv(fuse)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        # out = self.residual(self.gamma * (out_H + out_W) + x + y)

        return self.gamma * (out_H + out_W) + x + y


    
class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None
        self.mlp = conv_block(in_ch=dim, out_ch=dim)

    def forward(self, x):
        B, N, C = x.shape
        residual = x.reshape(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2)
        x_mlp = self.mlp(residual)
        residual = residual + x_mlp
        x_trans = residual.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x_trans = blk(x_trans)
        x_trans = x_trans.reshape(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2)
        out = x_trans + residual
        if self.upsample is not None:
            out = self.upsample(out)
        return out


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # H, W = self.input_resolution
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b c (h p1) (w p2)', p1=2, p2=2, c=C // 4)
        # x = x.view(B, -1, C // 4)
        # x = self.norm(x)

        return x

class Decoder(nn.Module):
    def __init__(self, channels, num_layers=4, img_size=1024, depths=[2, 2, 2], num_heads=[2, 4, 8]):
        super(Decoder, self).__init__()
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(num_layers):
            concat_linear = nn.Linear(2 * channels[i_layer], channels[i_layer])

            if i_layer == 0:
                layer_up = PatchExpand(dim=channels[i_layer], dim_scale=2)
            else:
                layer_up = BasicLayer_up(dim=channels[i_layer],
                                         input_resolution=(img_size // (2 ** (4 - i_layer + 1)),
                                                                 img_size // (2 ** (4 - i_layer + 1))),
                                         depth=depths[i_layer - 1],
                                         num_heads=num_heads[i_layer - 1],
                                         upsample=PatchExpand if (i_layer < (num_layers - 1)) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

    def forward(self, x, fused_feature):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, fused_feature[3 - inx]], 1)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes, img_size=512, num_layers=4):
        super(Net, self).__init__()
        channels = [512, 256, 128, 64, 32]
        # channels = [2048, 1024, 512, 256, 128, 64]
        # ------------------
        self.up4 = Up_Conv(channels[0], channels[1])
        self.conv4 = conv_block(channels[1], channels[1])
        self.up3 = Up_Conv(channels[1], channels[2])
        self.conv3 = conv_block(channels[2], channels[2])
        self.up2 = Up_Conv(channels[2], channels[3])
        self.conv2 = conv_block(channels[3], channels[3])
        self.up1 = Up_Conv(channels[3], channels[3])
        self.conv1 = conv_block(channels[3], channels[3])
        # ---------------------

        self.CNN_encoder_branch = ResNet_encoder(pretrained=True)

        # self.Transformer_encoder_branch = pvt_tiny_encoder()
        # self.Transformer_encoder_branch = pvt_small_encoder()
        # self.Transformer_encoder_branch = swin_tiny_encoder()
        self.Transformer_encoder_branch = swin_small_encoder()
        
        self.decoder = Decoder(channels=channels,
                               num_layers=4,
                               img_size=img_size,
                               depths=[2, 2, 2],
                               num_heads=[2, 4, 8])
        # self.CNN_decoder_branch = CNN_decoder(channels=channels, num_layers=4)

        self.Fusion1 = Fusion_block(in_dim=64)
        self.Fusion2 = Fusion_block(in_dim=128)
        self.Fusion3 = Fusion_block(in_dim=256)
        self.Fusion4 = Fusion_block(in_dim=512)

        self.up = Up_Conv(64, 64)
        self.conv = conv_block(128, 64)
        self.final1 = nn.Sequential(
            Conv(512, 256, 3, bn=True, relu=True),
            Conv(256, 128, 3, bn=True, relu=True),
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, bn=False, relu=False),
        )
        self.final3 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, bn=False, relu=False),
        )

    def forward(self, input):
        CNN_features = self.CNN_encoder_branch(input)
        Transformer_features = self.Transformer_encoder_branch(input)

        # fused_feature = []
        # x1 = self.Fusion1(Transformer_features[0], CNN_features[0])
        # x2 = self.Fusion2(Transformer_features[1], CNN_features[1])
        # x3 = self.Fusion3(Transformer_features[2], CNN_features[2])
        # x4 = self.Fusion4(Transformer_features[3], CNN_features[3])
        # x1 = CNN_features[0]
        # x2 = CNN_features[1]
        # x3 = CNN_features[2]
        # x4 = CNN_features[3]
        x1 = Transformer_features[0]
        x2 = Transformer_features[1]
        x3 = Transformer_features[2]
        x4 = Transformer_features[3]
        # fused_feature = [x1, x2, x3, x4]

        # -------------------------------
        # x4_up = self.up4(x4)
        # x4_up_c = torch.cat([x4_up, x3], dim=1)
        # x4_up_c_c = self.conv4(x4_up_c)
        #
        # x3_up = self.up3(x4_up_c_c)
        # x3_up_c = torch.cat([x3_up, x2], dim=1)
        # x3_up_c_c = self.conv3(x3_up_c)
        #
        # x2_up = self.up2(x3_up_c_c)
        # x2_up_c = torch.cat([x2_up, x1], dim=1)
        # x2_up_c_c = self.conv2(x2_up_c)
        #
        # x1_up = self.up1(x2_up_c_c)
        # x1_up_c = torch.cat([CNN_features[4], x1_up], dim=1)
        # x1_up_c_c = self.conv1(x1_up_c)
        #
        # x_trans = x2_up_c_c
        # x_final = x1_up_c_c
        # -------------------------------

        # -------------------------------
        x4_up = self.up4(x4)
        x4_up_c = self.conv4(x4_up)

        x3_up = self.up3(x4_up_c)
        x3_up_c = self.conv3(x3_up)

        x2_up = self.up2(x3_up_c)
        x2_up_c = self.conv2(x2_up)

        x1_up = self.up1(x2_up_c)
        x1_up_c = self.conv1(x1_up)

        x_trans = x2_up_c
        x_final = x1_up_c
        # -------------------------------


        # B, N, C = x_trans.shape
        # x_trans = x_trans.reshape(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2)

        # x_trans = self.decoder(x4, fused_feature)
        # decode_out = self.up(x_trans)
        # decode_out = torch.cat([CNN_features[4], decode_out], dim=1)
        # x_final = self.conv(decode_out)

        # x_final = torch.cat([CNN_features[4], decode_feature], dim=1)
        # x_final = self.conv(x_final)

        fusion_out = F.interpolate(self.final1(x4), scale_factor=32, mode='bilinear')
        decode_out = F.interpolate(self.final2(x_trans), scale_factor=4, mode='bilinear')
        final_out = F.interpolate(self.final3(x_final), scale_factor=2, mode='bilinear')


        return fusion_out, decode_out, final_out

# def draw_features(width, height, x, save_path):
#     fig = plt.figure(figsize=(16,16))
#     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
#     for i in range(width*height):
#         plt.subplot(height, width, i+1)
#         plt.axis('off')
#         img = x[0, i, :, :]
#         pmin = np.min(img)
#         pmax = np.max(img)
#         img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
#         img = img.astype(np.uint8)
#         img = cv.applyColorMap(img, cv.COLORMAP_JET)
#         img = img[:, :, ::-1]
#         plt.imshow(img)
#     fig.savefig(save_path, dpi=100)
#     fig.clf()
#     plt.close()


if __name__ == '__main__':
    net = Net(num_classes=1).cuda()
    x = torch.rand([4,3,512,512]).cuda()
    y, z, f = net(x)
    print(y.size())
    print(z.size())
    print(f.size())
