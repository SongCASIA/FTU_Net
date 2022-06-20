import torch
import torch.nn as nn

from network.swin_transformer import SwinTransformerBlock, PatchEmbed

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

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpand, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
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
        # if upsample is not None:
        #     self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        # else:
        #     self.upsample = None



    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x)
            # else:
            #     x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class Decoder(nn.Module):
    def __init__(self, img_size, in_ch, num_layers, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super(Decoder, self).__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=2, in_chans=in_ch, embed_dim=in_ch, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_layers = num_layers
        self.layers_up = nn.ModuleList()

        self.up4 = Up_Conv(in_ch, in_ch // 2)
        self.up3 = Up_Conv(in_ch // 2, in_ch // 4)
        self.up2 = Up_Conv(in_ch // 4, in_ch // 8)
        self.up1 = Up_Conv(channels[3], channels[3])
        for i_layer in range(4):
            layer_up = BasicLayer_up(dim=in_ch,
                                     input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                     depth=depths[self.num_layers - 1 - i_layer],
                                     num_heads=num_heads[self.num_layers - 1 - i_layer],
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     norm_layer=norm_layer)
