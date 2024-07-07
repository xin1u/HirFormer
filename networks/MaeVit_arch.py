      
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os,sys,time
curPath = os.path.abspath(os.path.dirname(__file__))
# print('curPath:', curPath)
rootPath = os.path.split(curPath)[0]
#rootPath = curPath
# print('rootPath:', rootPath)
sys.path.append(rootPath)
# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block# , PatchEmbed
# from models.Patch_embed import PatchEmbed
from networks.Patch_embed import PatchEmbed
# from utils.pos_embed import get_2d_sincos_pos_embed #丢掉位置编码
import torch.nn.functional as F


from functools import partial

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=4, ct_channels = 1):
        super().__init__()
        if num_scales==4:
            scales = (4, 8, 16, 32)
        elif num_scales==3:
            scales = (4, 8, 16)

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        #
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        #
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = []
        # for stage in self.stages:
        #     temp_fea = F.interpolate(input=stage(feats), size=(h, w), mode='nearest')
        #     print('temp_fea.shape:',temp_fea.shape)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=3,fea_chans=16,num_scales=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_residual = False):
        super().__init__()

        self.global_residual = global_residual
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)#img_size, 丢掉输入图片大小限制
        
        # num_patches = self.patch_embed.num_patches
        # num_patches = self.patch_embed.patch_size

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])  #Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)  因为qk_scale=None 参数而注释掉
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_for_unselected = nn.Linear(embed_dim, decoder_embed_dim, bias=True)


        #self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches , decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding ||   num_patches + 1  -->  num_patches 去掉了class token

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)]) #Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)   因为qk_scale=None 参数而注释掉

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * fea_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.pyramid_module = PyramidPooling(fea_chans, fea_chans, num_scales = num_scales , ct_channels = fea_chans // 4)
        self.last_conv =  nn.Conv2d(fea_chans, out_chans, kernel_size=3, padding=1, stride=1, bias=False)


    def initialize_weights(self):

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        #torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x, H, W):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = H // p
        w = W // p  # int(x.shape[1]**.5)
        assert h * w == x.shape[1]#验证unpatch后的h*w等于x的第二维度L

        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, w * p))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        #encoder_st1 = time.time()
        # x = torch.cat([x, mask], dim=1)
        #print('encoder---1: x',x.shape)
        x = self.patch_embed(x)
        #print('encoder---22: x', x.shape)
        #print('--------------time1:', time.time() - encoder_st1)
        #encoder_st2 = time.time()
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #print('--------------time2:', time.time() - encoder_st2)
        #print('encoder---333: x', x.shape)

        return x

    def forward_decoder(self, x):
        #这里的x应该是来自于上一步encoder中得到的latent codes
        # embed tokens
        x = self.decoder_embed(x)   #  nn.Linear(embed_dim, decoder_embed_dim) 这一步只是将之前encode中的embed_dim 数目映射时与  decoder_embed_dim 保持一致
        # add pos embed
        x = x   #+ self.decoder_pos_embed
        #print('decoder---1: x', x.shape)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        #print('decoder---22: x', x.shape)
        # predictor projection
        x = self.decoder_pred(x)

        #print('decoder---333: x', x.shape)

        return x

    def forward(self, imgs):
        _, _, ori_H, ori_W = imgs.size()
        #mask_in = mask_in.repeat(1, 3, 1, 1)

        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]

        pred_wOri_Size = self.unpatchify(pred, ori_H, ori_W )

        pred_wOri_Size = self.last_conv(self.pyramid_module(pred_wOri_Size))

        #print('pred_wOri_Size:', pred_wOri_Size.shape)
        if self.global_residual:
            pred_wOri_Size = pred_wOri_Size + imgs #self.patchify(imgs)

        return pred_wOri_Size  #, pred, mask_in, self.patch_embed.patch_size[0]



def mae_vit_small_patch16_dec128d4b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# 屡试不爽这个小模块 放在模型最后面整合多尺度features

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=4, ct_channels = 1):
        super().__init__()
        if num_scales==4:
            scales = (4, 8, 16, 32)
        elif num_scales==3:
            scales = (4, 8, 16)

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        #
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        #
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = []
        # for stage in self.stages:
        #     temp_fea = F.interpolate(input=stage(feats), size=(h, w), mode='nearest')
        #     print('temp_fea.shape:',temp_fea.shape)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))
    







# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


if __name__ == "__main__":
    model = mae_vit_small_patch16_dec128d4b(img_size= 256)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    # generator parameters: 5202496

    # base_channel=24
    # oriRDB: 8864100 ||   oriRDB1 (bais = False) : 8858052  || oriRDB1 (bais = True): 8865357
    # base_channel= 20
    #print(model)
    # count = 0
    # for name,module in model.named_modules():
    #     print(count,'-------------',name)
    #     count +=1
    # from functools import partial

    input = torch.randn(1, 3, 256, 256)
    mask  = torch.randn(1, 1, 256, 256)
    pred = model(input,mask)#, pred_wOri_Size, mask, _
    # #loss, pred, mask = model(input)
    print('-'*50)
    print(pred.shape)

    