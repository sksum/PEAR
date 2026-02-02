#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch,os,pickle
import torchvision
import torch.nn as nn
# from .styleunet import Blur,FusedLeakyReLU
from  copy import deepcopy
# from .embedder import get_embedder

import lightning as L
class DINO_Enocder(L.LightningModule):
    def __init__(self, output_dim=128,f2_residual=True):
        #multires=10,smplx_aseet_path="assets/SMPLX"
        super().__init__()
        
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.dino_model=self.dino_model.to('cpu')
        self.dino_normlize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.f2_residual=f2_residual
        in_dim = self.dino_model.blocks[0].attn.qkv.in_features
        hidden_dims=256
        out_dims=[256, 512, 1024, 1024]
        # modules
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_dim, out_dim, kernel_size=1, stride=1, padding=0,
            ) for out_dim in out_dims
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                out_dims[0], out_dims[0], kernel_size=4, stride=4, padding=0
            ),
            nn.ConvTranspose2d(
                out_dims[1], out_dims[1], kernel_size=2, stride=2, padding=0
            ),

            # ConvLayer_up(out_dims[0], out_dims[0], kernel_size=4, stride=4,padding=0),
            # ConvLayer_up( out_dims[1], out_dims[1], kernel_size=2, stride=2,padding=0),
            nn.Identity(),
            nn.Conv2d(out_dims[3], out_dims[3], kernel_size=3, stride=2, padding=1)
        ])
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(out_dims[0]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[1]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[2]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[3]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        ])

        self.refinenet = nn.ModuleList([
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
        ])
        self.output_conv = nn.Conv2d(hidden_dims, output_dim, kernel_size=3, stride=1, padding=1)

        # self.embedder,embed_dim= get_embedder(multires=multires)
        #self.uv_conv_layer=nn.Conv2d(768+embed_dim, 768, kernel_size=1, stride=1, padding=0)
        self.feamap_transformer_layer = nn.Sequential(deepcopy(self.dino_model.blocks[-1]),
                                                  deepcopy(self.dino_model.blocks[-1]))
        
        # with open(os.path.join(smplx_aseet_path,"smplx_template_position_map.pkl"), 'rb') as f:
        #     loaded_position_map = pickle.load(f)
        #self.template_position_map=torch.tensor(loaded_position_map)
        #position_map_embedded=self.embedder(self.template_position_map.reshape(-1,3)).reshape(self.template_position_map.shape[0],self.template_position_map.shape[1],-1)
        #position_map_embedded=position_map_embedded.permute(2,0,1).contiguous()[None]
        #self.register_parameter('position_map_embedded', nn.Parameter(position_map_embedded, requires_grad=False))
        
    def forward(self, images, output_size=None):
        batch_size=images.shape[0]
        images = self.dino_normlize(images)
        patch_h, patch_w = images.shape[-2]//14, images.shape[-1]//14
        
        image_features = self.dino_model.get_intermediate_layers(images, 4)
        out_features = []
        for i, feature in enumerate(image_features):
            feature = feature.permute(0, 2, 1).reshape(
                (feature.shape[0], feature.shape[-1], patch_h, patch_w)
            ).contiguous()
            feature = self.projects[i](feature)
            feature = self.resize_layers[i](feature)
            feature = torch.cat([
                    torchvision.transforms.functional.resize(images, (feature.shape[-2], feature.shape[-1]), antialias=True).detach(),
                    feature
                ], dim=1
            )
            out_features.append(feature)
        layer_rns = []
        for i, feature in enumerate(out_features):
            layer_rns.append(self.layer_rn[i](feature))

        path_4 = self.refinenet[0](layer_rns[3], size=layer_rns[2].shape[2:])
        path_3 = self.refinenet[1](path_4, layer_rns[2], size=layer_rns[1].shape[2:])
        path_2 = self.refinenet[2](path_3, layer_rns[1], size=layer_rns[0].shape[2:])
        path_1 = self.refinenet[3](path_2, layer_rns[0])
        out = self.output_conv(path_1)
        if output_size is not None:
            out = nn.functional.interpolate(out, output_size, mode="bilinear", align_corners=True)

        # out_uv=image_features[-3].reshape(batch_size,patch_w,patch_h,-1).permute(0,3,1,2).contiguous()
        # uv_pos_map=nn.functional.interpolate(self.position_map_embedded, out_uv.shape[2:], mode="bilinear", align_corners=True)
        # uv_pos_map=uv_pos_map.expand(batch_size,-1,-1,-1)
        # out_uv=torch.cat([out_uv,uv_pos_map],dim=1)
        # out_uv=self.uv_conv_layer(out_uv)
        # out_uv=out_uv.permute(0,2,3,1).reshape(batch_size,patch_h*patch_w,-1).contiguous()
        out_2=self.feamap_transformer_layer(image_features[-1])
        if self.f2_residual:
            out_2=out_2+image_features[-1]
        out_2=out_2.reshape(batch_size,patch_w,patch_h,-1).permute(0,3,1,2).contiguous()
        out_global = image_features[-1][:, 0]
        
        return out, out_2,out_global


class ResidualConvUnit(L.LightningModule):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(L.LightningModule):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output

# class ConvLayer_up(L.LightningModule):
#     def __init__(
#             self,
#             in_channel,
#             out_channel,
#             kernel_size,
#             stride=2,
#             blur_kernel=[1, 3, 3, 1],
#             padding=0,
#             bias=True,
#             activate=True,
#     ):
#         super().__init__()
#         self.weight = nn.Parameter(
#             torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
#         )
#         self.padding=padding
#         self.stride=int(stride)
#         factor = stride
#         p = (len(blur_kernel) - factor) - (kernel_size - 1)
#         pad0 = (p + 1) // 2 + factor - 1
#         pad1 = p // 2 + 1

#         self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
#         self.conv_t=nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,bias=bias)

#         if activate:
#             self.act=FusedLeakyReLU(out_channel, bias=bias)
        
#     def forward(self,input):

#         out = self.conv_t(input)
#         out = self.blur(out)
#         out=self.act(out)
#         return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DINO_Enocder(output_dim=128).to(device)

    batch_size = 2
    image_channels = 3
    image_height = 1024
    image_width = 1024
    fake_images = torch.randn(batch_size, image_channels, image_height, image_width).to(device)
    fake_images_s=torch.nn.functional.interpolate(fake_images,size=(14*52,14*52)) #w,h=14*x/(28)

    with torch.no_grad():
        output, uv_feature,global_feature = model(fake_images_s)
        print("Output shape:", output.shape)
        print("Global feature shape:", global_feature.shape)