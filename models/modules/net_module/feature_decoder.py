
#!/usr/bin/env python
import torch,pickle,os
import torch.nn as nn
# from .dino_encoder import ConvLayer_up
# from .styleunet import ConvBlock
from .codebook import Codebook
from .embedder import get_embedder
import numpy as np
import lightning as L
class UV_Feature_decoder(L.LightningModule):
    def __init__(self, output_dim=128,input_dim=768,code_shape=[32,32,512],multires=10,uv_up_factor=16,smplx_aseet_path="assets/SMPLX"):#16->32*16=512
        super().__init__()
        
        self.code_shape=code_shape
        self.embedder,embed_dim= get_embedder(multires=multires)
        laten_dim=code_shape[-1]
        
        self.conv_mapping=nn.Sequential(ConvBlock(input_dim,laten_dim,),
                                        nn.Conv2d(laten_dim,laten_dim,kernel_size=1,padding=0))
        num_codebook_vectors,code_latent_dim,code_beta=code_shape[0]*code_shape[1],laten_dim,1.0
        self.uv_codebook= Codebook(num_codebook_vectors=num_codebook_vectors,latent_dim=code_latent_dim,beta=code_beta)

        channels=[laten_dim,512,256]
        channels.extend([output_dim for i in range(6)])
        self.up_nums=int(np.log2(uv_up_factor))
        self.conv_ups_mapping=nn.ModuleList()
        self.conv_ups_mapping.append(nn.Conv2d(laten_dim,laten_dim,kernel_size=1,padding=0))
        for i in range (self.up_nums):
            self.conv_ups_mapping.append(
                #ConvLayer_up(channels[i],channels[i+1],kernel_size=3)
                nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2, padding=0),
            )
        self.conv_final=nn.Sequential(nn.Conv2d(channels[i+1]+embed_dim,channels[i+1],kernel_size=1,padding=0),nn.LeakyReLU(),
                                     ConvBlock(channels[i+1],channels[i+1],),
                                     )

        with open(os.path.join(smplx_aseet_path,"smplx_template_position_map.pkl"), 'rb') as f:
            loaded_position_map = pickle.load(f)
        self.template_position_map=torch.tensor(loaded_position_map)
        position_map_embedded=self.embedder(self.template_position_map.reshape(-1,3)).reshape(self.template_position_map.shape[0],self.template_position_map.shape[1],-1)
        position_map_embedded=position_map_embedded.permute(2,0,1).contiguous()[None]
        self.register_parameter('position_map_embedded', nn.Parameter(position_map_embedded, requires_grad=False))
    
    def forward(self,in_map):
        batch_size=in_map.shape[0]

        in_map=nn.functional.interpolate(in_map,(self.code_shape[0],self.code_shape[1]),mode='bilinear',align_corners=False)
        out_uv_map=self.conv_mapping(in_map)
        out_uv_map, min_encoding_indices, codebook_loss=self.uv_codebook(out_uv_map)
        out_uv_map=self.conv_ups_mapping[0](out_uv_map)   
        for i in range(self.up_nums):
            out_uv_map=self.conv_ups_mapping[i+1](out_uv_map)

        uv_pos_map=nn.functional.interpolate(self.position_map_embedded,(out_uv_map.shape[2],out_uv_map.shape[3]),mode='bilinear',align_corners=False).expand(batch_size,-1,-1,-1)
        
        out_uv_map=torch.cat([out_uv_map,uv_pos_map],dim=1)
        out_uv_map=self.conv_final(out_uv_map)
        
        return out_uv_map,codebook_loss

class Upsample_Feature_decoder(L.LightningModule):
    def __init__(self, output_dim=128,input_dim=768,up_factor=16):
        super().__init__()
        
        self.up_nums=int(np.log2(up_factor))
        self.conv_ups_mapping=nn.ModuleList()
        self.conv_fea_mapping=nn.ModuleList()
        self.skips_mapping=nn.ModuleList()
        channels=[input_dim,512,384,256,128,64]
        channels.extend([output_dim for i in range(6)])
        for i in range (self.up_nums):
            self.conv_ups_mapping.append(
                nn.Sequential(nn.ConvTranspose2d(channels[i]+3, channels[i+1], kernel_size=2, stride=2, padding=0),nn.LeakyReLU())
            )
            self.conv_fea_mapping.append(
                nn.Sequential(
                nn.Conv2d(channels[i+1]+3,channels[i+1],kernel_size=3,padding=1),nn.LeakyReLU(),
                nn.Conv2d(channels[i+1],channels[i+1],kernel_size=3,padding=1),nn.LeakyReLU()
                )
            )
            self.skips_mapping.append(nn.Conv2d(channels[i]+3, channels[i+1], 1, bias=False))
        self.conv_final=nn.Conv2d(channels[i+1],output_dim,kernel_size=3,stride=1,padding=1)
        
    
    def forward(self,in_map,images):
        
        batch_size=in_map.shape[0]
        image_resize_0=nn.functional.interpolate(images,(in_map.shape[2],in_map.shape[3]),mode='bilinear',align_corners=False)
        for i in range(self.up_nums):
            in_map_linear_up=nn.functional.interpolate(in_map,scale_factor=2,mode='bilinear',align_corners=False)
            in_map=self.conv_ups_mapping[i](torch.cat([in_map,image_resize_0],dim=1))
            image_resize_1=nn.functional.interpolate(images,(in_map.shape[2],in_map.shape[3]),mode='bilinear',align_corners=False)
            in_map=self.conv_fea_mapping[i](torch.cat([in_map,image_resize_1],dim=1))
            skip_map=self.skips_mapping[i](torch.cat([in_map_linear_up,image_resize_1],dim=1))
            image_resize_0=image_resize_1.clone()
            in_map=in_map+skip_map
        out_map=self.conv_final(in_map)
        return out_map    

class Vertex_GS_Decoder(L.LightningModule):
    # smplx vertices gaussian attributes predictor
    def __init__(self, in_dim=1024, dir_dim=27,color_out_dim=32,with_static_offset=False):
        super().__init__()
        self.with_static_offset = with_static_offset
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
        )
        layer_in_dim = in_dim//2 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, color_out_dim, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4, bias=True),
        )
        if self.with_static_offset:
            self.static_offset_layers = nn.Sequential(
                nn.Linear(layer_in_dim, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3, bias=True),
            )
            
    def forward(self, input_features, cam_dirs):
        input_features = self.feature_layers(input_features)
        cam_dirs = cam_dirs[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, cam_dirs], dim=-1)
        # color
        colors = self.color_layers(input_features)
        #colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        scales = torch.sigmoid(scales) * 0.05 #0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)
        
        res_dict={'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations,
                  'static_offsets':None}
        if self.with_static_offset:
            static_offsets = self.static_offset_layers(input_features)*0.05
            res_dict['static_offsets'] = static_offsets
        return  res_dict

class UP_Point_GS_Decoder(L.LightningModule):
    # Gaussian attributes predictor for unprojection points
    def __init__(self, in_dim=256, dir_dim=27,multires=8,color_out_dim = 27):
        super().__init__()
        color_out_dim = color_out_dim 
        opacity_out_dim= 1 
        scale_out_dim=3 
        rotation_out_dim=4 
        depth_out_dim=1
        lbs_out_dim=55 
        self.embedder,self.embed_dim= get_embedder(multires=multires)
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
        )
        
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, rotation_out_dim, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, scale_out_dim, kernel_size=1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, opacity_out_dim, kernel_size=1),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, color_out_dim, kernel_size=1),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, depth_out_dim, kernel_size=1),
        )
        self.lbs_weight_head = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),#+self.embed_dim
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//2, lbs_out_dim, kernel_size=1),
        )


    def forward(self, input_features,xy_image_coord,w2c,c2w,focal,cam_dirs,z_dir):
        #assume img_height=img_width
        b,h,w=input_features.shape[0],input_features.shape[2],input_features.shape[3]
        cam_dirs = cam_dirs[:, :, None, None].expand(-1, -1, h, w)
        input_features = torch.cat([input_features, cam_dirs], dim=1)
        gaussian_feature = self.feature_conv(input_features)
        # color
        colors = self.color_head(gaussian_feature)
        #colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_head(gaussian_feature)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_head(gaussian_feature)
        scales = torch.sigmoid(scales) * 0.05 #0.05
        # rotation
        rotations = self.rot_head(gaussian_feature)
        rotations = nn.functional.normalize(rotations)
        # depth
        depth_offest = self.depth_head(gaussian_feature)
        depth_offest =depth_offest*z_dir #*0.3? torch.sigmoid(
        
        results = {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations, 'depth_offest':depth_offest}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).contiguous()#.reshape(results[key].shape[0], -1, results[key].shape[1])

        xy_image_coord=nn.functional.interpolate(xy_image_coord[None].permute(0,3,1,2),(h,w),mode='bilinear',align_corners=False).permute(0,2,3,1).contiguous()[0]
        position=self.unproject_points(depth_offest,w2c,c2w,focal,xy_image_coord)
        results["positions"]=position
        # embedded_xyz=self.embedder(position.reshape(-1,3))
        # embedded_xyz=embedded_xyz.reshape(b,h,w,-1).permute(0,3,1,2).contiguous()
        # lbs_weights=self.lbs_weight_head(torch.cat([gaussian_feature,embedded_xyz],dim=1))
        lbs_weights=self.lbs_weight_head(gaussian_feature)
        # lbs_weights=torch.exp(10 * lbs_weights.permute(0, 2, 3, 1)).contiguous()
        # lbs_weights=lbs_weights/torch.sum(lbs_weights, dim=-1, keepdim=True)
        lbs_weights=torch.softmax(10 * lbs_weights.permute(0, 2, 3, 1).contiguous(), dim=-1)
        results["lbs_weights"]=lbs_weights
        
        return results
    
    def unproject_points(self,z_depth_offest,w2c,c2w,focal,xy_image_coord):
        # unproject points from 2D to 3D based on depth

        batch_size=w2c.shape[0]
        z_depth_offest=z_depth_offest.permute(0,2,3,1).contiguous()
        xy_image_coord=xy_image_coord[None].expand(batch_size,-1,-1,-1)
        depth_refer=(w2c[:,2,3])[:,None,None,None].expand(-1,xy_image_coord.shape[1],xy_image_coord.shape[2],-1)
        
        z_depth=z_depth_offest+depth_refer
        xy_cam_coord=(xy_image_coord*z_depth)/focal
        z_cam_coord=z_depth
        xyz_cam_coord=torch.cat([xy_cam_coord,z_cam_coord],dim=-1)#b h w 3
        xyz_world_coord=torch.einsum('bij,bhwj->bhwi',c2w[:,:3,:3],xyz_cam_coord)
        xyz_world_coord=xyz_world_coord+c2w[:,None,None,:3,3]
        return xyz_world_coord
    
    # def unproject_points_pytorch3d(self,z_depth_offest,w2c,c2w,focal,xy_image_coord):
    #     h,w=xy_image_coord.shape[1],xy_image_coord.shape[2]
        
    # def unproject_points_2(self,z_depth_offest,w2c,c2w,focal,xy_image_coord):
    #     # unproject points through image plane 

    #     batch_size=w2c.shape[0]
    #     z_depth_offest=z_depth_offest.permute(0,2,3,1).contiguous()
    #     plane_size=xy_image_coord.shape[1]
    #     device=w2c.device
        
    #     x, y = torch.meshgrid(
    #     torch.linspace(1, -1, plane_size, dtype=torch.float32,device=device), 
    #     torch.linspace(1, -1, plane_size, dtype=torch.float32,device=device), 
    #     indexing="xy",)
    #     w2c=w2c.clone()
    #     c2c_mat=torch.tensor([[-1, 0, 0, 0],
    #                           [ 0,-1, 0, 0],
    #                           [ 0, 0, 1, 0],
    #                           [ 0, 0, 0, 1],
    #                           ],dtype=torch.float32,device=device)[None].expand(batch_size,-1,-1)
    #     transforms=torch.matmul(c2c_mat,w2c)
    #     R = transforms[:,:3, :3]; T = transforms[:,:3, 3:]
    #     cam_dirs = torch.tensor([[0., 0., 1.]], dtype=torch.float32,device=device).expand(batch_size,-1)
    #     ray_dirs = torch.nn.functional.pad(
    #         torch.stack([x/focal, y/focal], dim=-1), (0, 1), value=1.0
    #     )[None].expand(batch_size,-1,-1,-1)
        
    #     cam_dirs = torch.einsum("brc,bc->br",R,cam_dirs)#torch.matmul(R, cam_dirs.reshape(batch_size,-1, 3)[:, :, None])[..., 0]
    #     ray_dirs = torch.einsum("brc,bhwc->bhwr",R,ray_dirs)#torch.matmul(R, ray_dirs.reshape(batch_size,-1, 3)[:, :, None])[..., 0]
    #     origins = (-torch.matmul(R, T)[..., 0])#[:,None,None,:].expand(-1,plane_size,plane_size,-1)
    #     distance = torch.einsum("br,br->b",origins,cam_dirs).abs()[:,None,None,None]#((origins[0] * cam_dirs[0]).sum()).abs()
    #     plane_points = origins[:,None,None,:] + (distance+z_depth_offest) * ray_dirs
        
    #     return plane_points

class UV_Point_GS_Decoder(L.LightningModule):
    # Gaussian attributes predictor for uv points
    def __init__(self, in_dim=128, dir_dim=27,multires=8,color_out_dim = 27):
        super().__init__()
        color_out_dim = color_out_dim 
        opacity_out_dim= 1 
        scale_out_dim=3 
        rotation_out_dim=4 
        local_pos_dim=3
        hid_dim_1=max(in_dim,128)
        hid_dim_2=max(in_dim//2,64)
        self.embedder,self.embed_dim= get_embedder(multires=multires)
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, hid_dim_1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, stride=1, padding=1),
        )
        
        self.rot_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, rotation_out_dim, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, scale_out_dim, kernel_size=1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, opacity_out_dim, kernel_size=1),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, color_out_dim, kernel_size=1),
        )
        self.local_pos_head = nn.Sequential(
            nn.Conv2d(hid_dim_1, hid_dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_1, hid_dim_2, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hid_dim_2, local_pos_dim, kernel_size=1),
        )


    def forward(self, input_features,cam_dirs):
        #assume img_height=img_width
        b,h,w=input_features.shape[0],input_features.shape[2],input_features.shape[3]
        cam_dirs = cam_dirs[:, :, None, None].expand(-1, -1, h, w)
        input_features = torch.cat([input_features, cam_dirs], dim=1)
        gaussian_feature = self.feature_conv(input_features)
        # color
        colors = self.color_head(gaussian_feature)
        # opacity
        opacities = self.opacity_head(gaussian_feature)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_head(gaussian_feature)
        scales = torch.exp(scales)  #0.05 * 0.05
        # rotation
        rotations = self.rot_head(gaussian_feature)
        rotations = nn.functional.normalize(rotations)
        # local position
        local_pos = self.local_pos_head(gaussian_feature)
        
        results = {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations, 'local_pos':local_pos}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).contiguous()#.reshape(results[key].shape[0], -1, results[key].shape[1])
        
        return results
    
class Vertex_offset_Decoder(L.LightningModule):
    # static offset predictor for smplx vertices
    def __init__(self, in_dim=768,multires=8,out_dim = 3):
        super().__init__()
        self.embedder,self.embed_dim= get_embedder(multires=multires)
        hid_dim_1=256
        hid_dim_2=128
        in_dim=in_dim+self.embed_dim
        self.static_offset=nn.Sequential(Res_MLP_Block(in_dim,hid_dim_1),
                                         Res_MLP_Block(hid_dim_1,hid_dim_2),
                                         )
        self.offset_head = nn.Linear(hid_dim_2, out_dim)
    def forward(self, xyz, input_features):
        batch_size=input_features.shape[0]
        embedded_xyz=self.embedder(xyz)
        embedded_xyz=embedded_xyz[None].expand(batch_size,-1,-1)
        input_features=torch.cat([input_features,embedded_xyz],dim=-1)
        static_offset=self.static_offset(input_features)
        offset=self.offset_head(static_offset)*0.05
        
        return offset

class Res_MLP_Block(L.LightningModule):
    def __init__(self, in_dim, out_dim,mid_layers=2):
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.LeakyReLU(inplace=True),
            *[nn.Sequential(nn.Linear(out_dim, out_dim, bias=True),
                           nn.LeakyReLU(inplace=True)) for _ in range(mid_layers)]
        )
        self.mlp_skip = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, input):
        return self.mlps(input) + self.mlp_skip(input)
    
class ConvBlock(L.LightningModule):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3,  padding=1)
        self.act=nn.LeakyReLU(inplace=True)
    def forward(self, input):
        out = self.conv1(input)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        return out

if __name__ =="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UV_Feature_decoder().to(device)

    batch_size = 4
    input_channels = 768
    height = 54
    width = 54
    dummy_input = torch.randn(batch_size, input_channels, height, width).to(device)
    

    with torch.no_grad(): 
        output, codebook_loss = model(dummy_input)
    
    import pdb
    pdb.set_trace()
    print("Output shape:", output.shape)
    print("Codebook loss:", codebook_loss.item())
        

        
        
        