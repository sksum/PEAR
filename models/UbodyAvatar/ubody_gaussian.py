import math,time
import torch,os
import torchvision
import torch.nn as nn
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding
from pytorch3d.renderer import PerspectiveCameras
from ..modules.ehm import EHM_v2 
from ..modules.smplx import SMPLX
from..modules.net_module.dino_encoder_h import  DINO_Enocder
# from..modules.net_module.dino_encoder import  DINO_Enocder
from ..modules.net_module.feature_decoder import UV_Feature_decoder,Upsample_Feature_decoder,Vertex_GS_Decoder,UV_Point_GS_Decoder,Vertex_offset_Decoder
from ..modules.net_module.styleunet.styleunet_2 import StyleUNet
from utils.graphics_utils import BaseMeshRenderer
from utils.render_nvdiffrast import NVDiffRenderer
from roma import rotmat_to_unitquat, quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw,unitquat_to_rotmat
import lightning as L
from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
from utils.graphics_utils import compute_face_orientation
from utils.general_utils import inverse_sigmoid
#infer ubody gaussians from image and smplx

###uvmap_binding-ehmv2
class Ubody_Gaussian_inferer(L.LightningModule):
    def __init__(self, cfg):
        super( ).__init__()
        self.cfg=cfg
        # if not cfg.with_neural_refiner: 
        #     assert cfg.color_dim==(cfg.sh_degree+1)**2*3
        
        #self.device=device
        self.uvmap_size=cfg.uvmap_size
        #albation study
        self.with_static_offset=cfg.with_static_offset
        self.with_head_feature=cfg.with_head_feature
        self.with_smplx_tracking=cfg.with_smplx_tracking
        self.with_inverse_mapping=cfg.with_inverse_mapping
        self.with_uv_gaussian=cfg.with_uv_gaussian
        self.with_smplx_gaussian=cfg.with_smplx_gaussian
        
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        self.intri_cam={'focal':cfg.invtanfov,'size': [cfg.image_size, cfg.image_size]}
        xy_image_coord=(get_pixel_coordinates(cfg.image_size,cfg.image_size)-0.5*cfg.image_size)/(cfg.image_size*0.5)# [-1,1]
        self.xy_image_coord=nn.Parameter(xy_image_coord, requires_grad=False)
        sample_out_dim=cfg.prj_out_dim #vertex feature dim
        
        #general feature extrator
        self.dino_encoder=DINO_Enocder(output_dim=cfg.dino_out_dim,output_dim_2=sample_out_dim,hidden_dims=sample_out_dim//2,
                                       with_head_feature=self.with_head_feature,)#.to(device)f2_residual=(cfg.vertex_smaple_type=="projection"
        for param in self.dino_encoder.dino_model.parameters():
            param.requires_grad = False
        if self.with_head_feature:
            self.head_feature_fuse=nn.ModuleList()
            self.head_feature_fuse.append(nn.Sequential(nn.Linear(cfg.global_vertex_dim*2,cfg.global_vertex_dim)))
            self.head_feature_fuse.append(nn.Sequential(nn.Linear(sample_out_dim*2,sample_out_dim)))
            self.head_feature_fuse.append(nn.Sequential(nn.Linear(cfg.dino_out_dim*2,cfg.dino_out_dim)))
        #vertex feature decoder
        # if self.cfg.vertex_smaple_type=="projection":
        #     self.prj_feature_decocer = Upsample_Feature_decoder(output_dim=cfg.prj_out_dim)
        
        self.global_feature_mapping=nn.Sequential(nn.Linear(768,cfg.global_vertex_dim),nn.LeakyReLU(inplace=True),nn.Linear(cfg.global_vertex_dim,cfg.global_vertex_dim),
                                                  nn.LeakyReLU(inplace=True),nn.Linear(cfg.global_vertex_dim,cfg.global_vertex_dim))
        if self.with_smplx_gaussian:
            self.vertex_gs_decoder = Vertex_GS_Decoder(in_dim=sample_out_dim+cfg.smplx_fea_dim+cfg.global_vertex_dim,dir_dim=self.direnc_dim,color_out_dim=cfg.color_dim,
                                                    with_static_offset=self.with_static_offset)#.to(device)
        
        # self.vertex_offset_decoder = Vertex_offset_Decoder(in_dim=512+cfg.smplx_fea_dim,)
        # self.global_feature_mapping_2=nn.Sequential(nn.Linear(768,512),nn.LeakyReLU(inplace=True),nn.Linear(512,512),nn.LeakyReLU(inplace=True),nn.Linear(512,512))
        #self.nvd_render=NVDiffRenderer(lighting_type='constant',focal_length=cfg.invtanfov,image_size=(512,512))

        if self.with_smplx_tracking: # ablation study
            self.smplx=SMPLX( cfg.smplx_assets_dir, add_teeth=cfg.add_teeth, uv_size=self.uvmap_size)#.to(device)
            self.ehm =self.smplx
        else:
            self.ehm =  EHM_v2( cfg.flame_assets_dir, cfg.smplx_assets_dir, cfg.mano_assets_dir,  uv_size = self.uvmap_size)#.to(device)
            self.smplx = self.ehm.smplx
            
        self.v_template=torch.nn.Parameter(self.ehm.v_template,requires_grad=False)
        self.laplacian_matrix=torch.nn.Parameter(self.ehm.laplacian_matrix,requires_grad=False)
        
        num_vertices=self.smplx.v_template.shape[0]
        self.vertex_base_feature = nn.Parameter(torch.randn(num_vertices, cfg.smplx_fea_dim), requires_grad=True,)#device=device
        
        #uv point feature decoder
        self.uv_feature_decoder = StyleUNet(in_size=self.uvmap_size, out_size=self.uvmap_size,activation=False,in_dim=cfg.dino_out_dim+3, out_dim=cfg.uv_out_dim,extra_style_dim=512)
        self.uv_style_mapping=nn.Sequential(nn.Linear(768,512),nn.LeakyReLU(inplace=True),nn.Linear(512,512),nn.LeakyReLU(inplace=True),nn.Linear(512,512))
        
        self.uv_base_feature = nn.Parameter(torch.randn((32,self.uvmap_size,self.uvmap_size)), requires_grad=True,)
        self.uv_point_decoder = UV_Point_GS_Decoder(in_dim=cfg.uv_out_dim + 32, dir_dim=self.direnc_dim,color_out_dim=cfg.color_dim)#.to(device)
        self.mesh_renderer = BaseMeshRenderer(faces=self.smplx.faces_tensor,image_size=512,faces_uvs=self.smplx.faces_uv_idx,
                                       verts_uvs=self.smplx.texcoords,lbs_weights=self.smplx.lbs_weights,focal_length=self.cfg.invtanfov)
        self.uv_mask_flat=self.smplx.uvmap_mask.flatten()
           
    def sample_uv_feature(self,uv_coord,uv_feature_map):
        #uv_feature_map b c h w
        batch_size=uv_feature_map.shape[0]
        grid = uv_coord.clone()
        grid[..., 0] = 2.0 * uv_coord[..., 0] - 1.0  # u
        grid[..., 1] = 2.0 * uv_coord[..., 1] - 1.0  # v
        grid = grid[None,None].expand(batch_size,-1,-1,-1) #b 1 n 2
        sampled_features = nn.functional.grid_sample(uv_feature_map, grid, mode='bilinear', padding_mode='border', align_corners=False)
        sampled_features = sampled_features.squeeze(-2).permute(0,2,1).contiguous() #b n c
        return sampled_features
    
    def sample_prj_feature(self,vertices,feature_map,w2c_cam,vertices_img=None):
        batch_size=feature_map.shape[0]
        if vertices_img is None:
            vertices_homo=torch.cat([vertices,torch.ones_like(vertices[:,:,:1])],dim=-1)
            vertices_cam=torch.einsum('bij,bnj->bni',w2c_cam,vertices_homo)[:,:,:3]
            vertices_img=vertices_cam*self.cfg.invtanfov/(vertices_cam[:,:,2:]+1e-7)
        sampled_features = nn.functional.grid_sample(feature_map, vertices_img[:,None,:,:2], mode='bilinear', padding_mode='border', align_corners=False)
        sampled_features = sampled_features.squeeze(-2).permute(0,2,1).contiguous() #b n c
        return sampled_features,vertices_img
    # 
    # 根据 uv map的像素与三角面片的对应关系，可以得到uv map 像素对应的 3D 坐标，然后用3D坐标 在DINO估计的图像特征插值得到 UV feature
    def convert_pixel_feature_to_uv(self,img_features,deformed_vertices,w2c_cam,visble_faces=None,uv_features=None,head_features=None,head_transform=None):
        batch_size,feature_dim = img_features.shape[0],img_features.shape[1]
        if uv_features is None: # [B,3+32,512,512]
            uv_features=torch.zeros((batch_size,feature_dim,self.cfg.uvmap_size,self.cfg.uvmap_size),device=img_features.device,dtype=torch.float32)
        # TODO: 这一块好像是通用的，没必要写在这里
        uvmap_f_idx=self.smplx.uvmap_f_idx  # [512,512] 获取 UV 图中每个像素对应的三角面和重建顶点坐标
        uvmap_f_mask=self.smplx.uvmap_mask
        uvmap_f_bary=self.smplx.uvmap_f_bary[None].expand(batch_size,-1,-1,-1) # [512,512, 3]  每个像素在该三角面上的重心坐标，也就是每个点坐标所占比例
        faces=self.smplx.faces_tensor # [N,3] n个面，每个面对应3个顶点
        uv_vertex_id=faces[uvmap_f_idx] # H W 3  每个uv pixel 对应的3个顶点
        
        uv_vertex=deformed_vertices.permute(1,2,0).contiguous()[uv_vertex_id] # H W 3 3 B  3个顶点的3个坐标
        uv_vertex=uv_vertex.permute(4,0,1,2,3).contiguous()# B H W k 3
        uv_vertex= torch.einsum('bhwk,bhwkn->bhwn',uvmap_f_bary,uv_vertex)# B H W 3  # 每个顶点加权得到真正的坐标
        uv_vertex_homo = torch.cat([uv_vertex,torch.ones_like(uv_vertex[:,:,:,:1])],dim=-1)
        uv_vertex_cam = torch.einsum('bij,bhwj->bhwi',w2c_cam,uv_vertex_homo)[:,:,:,:3] # 映射到相机坐标系
        vertices_img = uv_vertex_cam * self.cfg.invtanfov/(uv_vertex_cam[...,2:]+1e-7) # [B,h,w,3]
        uv_features = nn.functional.grid_sample(img_features, vertices_img[:,:,:,:2], mode='bilinear', padding_mode='zeros', align_corners=False) #从图像中采样对应位置的特征，得到初步的 UV 特征图
        mask=self.smplx.uvmap_mask.clone()[None].repeat(batch_size,1,1)
        if self.with_head_feature:
            H,W=vertices_img.shape[1],vertices_img.shape[2]
            head_vertices_img=vertices_img[:,:,:,:2].reshape(batch_size,-1,2)
            head_vertices_img=head_vertices_img[:,self.ehm.head_idxs_uv_flat]
            head_vertices_img[...,0]=(head_vertices_img[...,0]-head_transform[:,None,2])*head_transform[:,None,0]
            head_vertices_img[...,1]=(head_vertices_img[...,1]-head_transform[:,None,3])*head_transform[:,None,1]
            head_uv_feature=nn.functional.grid_sample(head_features, head_vertices_img[:,None,:,:2], mode='bilinear', padding_mode='zeros', align_corners=False)
            head_uv_feature=head_uv_feature.squeeze(-2).permute(0,2,1).contiguous() #b n c
            uv_features=uv_features.permute(0,2,3,1).reshape(batch_size,H*W,-1).contiguous().clone() #b h*w c
            head_uv_feature_rgb=head_uv_feature[...,:3]
            head_uv_feature_lat=head_uv_feature[...,3:]
            uv_features_rgb=uv_features[:,:,:3].clone()
            uv_features_lat=uv_features[:,:,3:].clone()
            uv_features_rgb[:,self.ehm.head_idxs_uv_flat]=head_uv_feature_rgb

            uv_features_lat[:,self.ehm.head_idxs_uv_flat]=self.head_feature_fuse[2](torch.cat([uv_features_lat[:,self.ehm.head_idxs_uv_flat],head_uv_feature_lat],dim=-1))
            uv_features=torch.cat([uv_features_rgb,uv_features_lat],dim=-1)
            uv_features=uv_features.reshape(batch_size,H,W,-1).permute(0,3,1,2).contiguous()
            
        if visble_faces is not None: # 可见面片遮罩处理（visible_faces）
            num_faces=self.smplx.faces_tensor.shape[0]
            f_offset=torch.arange(batch_size,device=uvmap_f_idx.device,dtype=torch.int32)*num_faces
            
            # b_uvmap_f_idx=uvmap_f_idx[None].repeat(batch_size,1,1)
            # b_uvmap_f_idx=b_uvmap_f_idx+f_offset[:,None,None,]*uvmap_f_mask[None]
            # visble_mask=torch.isin(b_uvmap_f_idx,torch.unique(visble_faces))
            # or
            all_faces=torch.arange(0,faces.shape[0],device=uvmap_f_idx.device,dtype=torch.int32)
            all_faces=all_faces[None].repeat(batch_size,1)
            all_faces=all_faces+f_offset[:,None]
            visble_all_faces=torch.isin(all_faces,torch.unique(visble_faces))
            visble_mask=visble_all_faces[:,uvmap_f_idx]
            
            # print(time.time()-start_time)
            mask=mask*visble_mask
        #save_visual_pixel_to_uv(uv_features[0,:3],img_features[0,:3],self.smplx.faces_uv_idx.cpu().numpy(),self.smplx.texcoords.cpu().numpy(),name='visual_pixel_to_uv_nomasked')
        uv_features=uv_features*mask[:,None]
        # save_visual_pixel_to_uv(uv_features[0,:3],img_features[0,:3],self.smplx.faces_uv_idx.cpu().numpy(),self.smplx.texcoords.cpu().numpy(),name='visual_pixel_to_uv_mask')
        # import ipdb;ipdb.set_trace()
        return uv_features
        
    def forward(self,batch, pd_smplx_dict, pd_w2c_cam=None):
        try:
            batch_size = batch['image'].shape[0]
        except:
            batch_size = pd_w2c_cam.shape[0]
        extra_dict={}
        if 'head_transform' not in batch.keys():
            batch['head_transform']=None
            
        head_images = batch['head_image'] if self.with_head_feature else None

        batch['image'] = batch['image'] * batch['mask']  # 将 mask 后的图再求纹理,  518

        # 为啥他这里是 先进行 mask 然后进行 normalization  
        dino_feature_dict = self.dino_encoder( batch['image'], output_size=self.cfg.image_size, head_images=head_images)
        img_feature, img_feature_2, global_faeature = dino_feature_dict['f_map1'], \
                    dino_feature_dict['f_map2'],dino_feature_dict['f_global'] # [B,32,512,512]  [B,128,512,512]  [B, 768] 

        vertex_global_feature = self.global_feature_mapping(global_faeature)  # 3层的 MLP  [B,786] -> [B,256]
        head_img_feature=None
        if pd_w2c_cam is None: # For debug
            pd_w2c_cam = batch["w2c_cam"]
        cam_dirs = get_cam_dirs(pd_w2c_cam)
        cam_dirs = self.harmo_encoder(cam_dirs) # [B,27]
        vertex_base_feature = self.vertex_base_feature[None].expand(batch_size,-1,-1) # [B,10595,128] learnable params
        
        # vertex_global_feature_2=self.global_feature_mapping_2(global_faeature)
        # vertex_global_feature_2=vertex_global_feature_2[:,None,:].expand(-1,self.vertex_base_feature.shape[-2],-1)
        # static_offset=self.vertex_offset_decoder(self.v_template,torch.cat([vertex_global_feature_2,vertex_base_feature],dim=-1))
        if pd_smplx_dict is not None :
            self.smplx_deform_res = pd_smplx_dict
        else:
            if self.with_smplx_tracking:  # False
                self.smplx_deform_res=self.smplx(batch['smplx_coeffs'])
            else:
                self.smplx_deform_res=self.ehm(batch['smplx_coeffs'],batch['flame_coeffs']) #  可以复用 , static_offset=static_offset
                
        # if self.cfg.vertex_smaple_type=="projection":
        #     prj_feature=self.prj_feature_decocer(img_feature_2,batch['image'])
        # 第二个特征 img_feature_2  用来估计 vertex gs
        vertex_sample_feature, vertex_prj = self.sample_prj_feature(self.smplx_deform_res['vertices'],img_feature_2,pd_w2c_cam) # [B,10595,128]   [B,10595,3] 
        vertex_global_feature=vertex_global_feature[:,None,:].expand(-1,vertex_sample_feature.shape[-2],-1) # [B,10595,256]  

        if self.with_head_feature : # False
            head_img_feature,head_img_feature2=dino_feature_dict['head_f_map1'],dino_feature_dict['head_f_map2']
            head_img_feature=torch.cat([head_images,head_img_feature],dim=1)
            head_global_feature=dino_feature_dict['head_f_global']
            vertex_head_global_feature=self.global_feature_mapping(head_global_feature)
                  
            head_vertex_prj=vertex_prj[:,self.ehm.head_idxs_temp]
            head_vertex_prj[...,0]=(head_vertex_prj[...,0]-batch['head_transform'][:,None,2])*batch['head_transform'][:,None,0]
            head_vertex_prj[...,1]=(head_vertex_prj[...,1]-batch['head_transform'][:,None,3])*batch['head_transform'][:,None,1]
            head_vertex_sample_feature,head_vertex_prj=self.sample_prj_feature(None,head_img_feature2,None,vertices_img=head_vertex_prj)
            fuse_head_smaple_feature=self.head_feature_fuse[1](torch.cat([vertex_sample_feature[:,self.ehm.head_idxs_temp],head_vertex_sample_feature],dim=-1))
            fuse_head_global_feature=self.head_feature_fuse[0](torch.cat([vertex_global_feature[:,0],vertex_head_global_feature],dim=-1))
            vertex_sample_feature,vertex_global_feature=vertex_sample_feature.clone(),vertex_global_feature.clone()
            vertex_sample_feature[:,self.ehm.head_idxs_temp]=fuse_head_smaple_feature
            vertex_global_feature[:,self.ehm.head_idxs_temp]=fuse_head_global_feature[:,None,:].expand(-1,self.ehm.head_idxs_temp.shape[0],-1)
        
        if self.with_smplx_gaussian:      # True
            # 每个样本的 img_feature_2 下的采样特征， 所有样本共享的可学习基础特征， 每个样本内所有顶点共享的全局特征
            vertex_sample_feature=torch.cat([vertex_sample_feature,vertex_base_feature,vertex_global_feature],dim=-1) # [B,10595,128]  [B,10595,128]  [B,10595,256]    
            vertex_gs_dict = self.vertex_gs_decoder(vertex_sample_feature,cam_dirs)  # 多头 浅层 MLP 解码 出属性（r，s，o，c 以及一个特征）
            vertex_gs_dict["positions"] = self.v_template.clone()[None].expand(batch_size,-1,-1) # 换成 temple 的坐标

        else:
            tdevice=self.v_template.device
            vertex_gs_dict={'scales':torch.ones((batch_size,1,3),device=tdevice),'rotations':torch.ones((batch_size,1,4),device=tdevice),
                              'opacities':torch.zeros((batch_size,1,1),device=tdevice),'positions':torch.ones((batch_size,1,3),device=tdevice),
                              'colors':torch.ones((batch_size,1,32),device=tdevice),'static_offsets':None}
        
        # uvmap gs
        if self.with_uv_gaussian:  # True
            # 518 插值成 512
            image_rgb =nn.functional.interpolate(batch['image'],(self.cfg.image_size,self.cfg.image_size),mode='bilinear',align_corners=False) 
            img_feature=torch.cat([image_rgb, img_feature],dim=1)  # 把第一个特征与图像拼在一起，第一个特征用来估计 uv gs
            if self.with_inverse_mapping:
                with torch.no_grad():
                    # [B,512,512,1] 将渲染， visble_faces指代像素 被哪个三角形 渲染， -1 代表没被渲染
                    visble_faces, fragments = self.mesh_renderer.render_fragments(self.smplx_deform_res['vertices'], transform_matrix=pd_w2c_cam)
                # head_transform， head_img_feature= None  [B,35,512,512]
                uvmap_features = self.convert_pixel_feature_to_uv(img_feature,self.smplx_deform_res['vertices'],pd_w2c_cam,visble_faces=visble_faces,
                                                                head_features=head_img_feature,head_transform=batch['head_transform']) # None
            else:
                uvmap_features = img_feature
            # textured_mesh_image=self.mesh_renderer.render_textured_mesh(self.smplx_deform_res['vertices'],uvmap_features[:,:3].permute(0,2,3,1).contiguous(),
            #                                                             fragments=fragments,transform_matrix=batch['w2c_cam'])
            # textured_mesh_image=self.nvd_render.render_rgba(self.smplx_deform_res['vertices'], self.smplx.faces_tensor, self.smplx.texcoords, self.smplx.faces_uv_idx, 
            #                                                 uvmap_features[:,:3].permute(0,2,3,1).contiguous(),transform_matrix=batch['w2c_cam'])
            # save_visual_images(textured_mesh_image.permute(0,3,1,2)[0,:3],'z_temp/textured_mesh_image_head.png') # ['rgba']
            
            extra_style = self.uv_style_mapping(global_faeature) # 全局特征 估计 MLP  [768 ->  512 ]
            uvmap_features=self.uv_feature_decoder(uvmap_features,extra_style=extra_style)
            uvmap_features=torch.cat([uvmap_features,self.uv_base_feature[None].expand(batch_size,-1,-1,-1)],dim=1) # [B, 128, 512, 512]
            # 卷积层  [B, h, w, n] 将特征和方向加进去再 多头卷积 解码 得到 uv gs，  c, s, r, o, p
            uv_point_gs_dict=self.uv_point_decoder(uvmap_features,cam_dirs)  # 

            # 这段代码的核心目标是将与 UV 图有关的数据，从全图中提取出落在 mesh 表面的有效部分，并准备好面片索引、重心坐标和纹理信息用于后续操作（如纹理采样、重建等）。
            # uv_point_gs_dict['opacity_map']=uv_point_gs_dict['opacities']
            for key in uv_point_gs_dict.keys():
                gs_f0=uv_point_gs_dict[key].reshape(batch_size,self.uvmap_size*self.uvmap_size,-1)  # 铺开做 mask
                uv_point_gs_dict[key] =gs_f0[:,self.uv_mask_flat,:]  # 这里为啥要加一个 uv mask

            binding_face=self.smplx.uvmap_f_idx.clone().reshape(1,self.uvmap_size*self.uvmap_size,-1) # uvmap_f_idx [512,512]
            uv_point_gs_dict['binding_face']=binding_face[:,self.uv_mask_flat,:] #  # 用于知道这个 UV 点对应哪一个 mesh face。
            face_bary=self.smplx.uvmap_f_bary.clone().reshape(1,self.uvmap_size*self.uvmap_size,-1)
            uv_point_gs_dict['face_bary']=face_bary[:,self.uv_mask_flat,:] #  是每个 UV 像素对应在三角面片三个顶点的比重 ，
            extra_dict['uvmap_texture']=torch.sigmoid(uvmap_features[:,:3].permute(0,2,3,1).contiguous()) # [B, 512, 512, 3]

        else:
            uv_point_gs_dict={'scales':torch.ones((batch_size,1,3)),'rotations':torch.ones((batch_size,1,4)),
                              'opacities':torch.zeros((batch_size,1,1)),'local_pos':torch.ones((batch_size,1,3)),
                              'colors':torch.ones((batch_size,1,32)),'binding_face':torch.ones((batch_size,1,1)),
                              'face_bary':torch.ones((batch_size,1,3))}
            
        return vertex_gs_dict, uv_point_gs_dict, extra_dict

# deform ubody gaussians from canonical space to deformed space
class Ubody_Gaussian(L.LightningModule):
    def __init__(self,cfg,vertex_gaussian_assets,uv_gaussian_assets,pruning=False):
        super().__init__()
        # self.device=device
        self.cfg=cfg
        
        #ablation study
        self.with_smplx_tracking=cfg.with_smplx_tracking
        self.with_uv_gaussian=cfg.with_uv_gaussian
        self.with_smplx_gaussian=cfg.with_smplx_gaussian
        self.with_neural_refiner=cfg.with_neural_refiner
        
        self.max_sh_degree=cfg.sh_degree
        self.opacity_threshold=cfg.opacity_threshold
        
        self._smplx_scaling=vertex_gaussian_assets['scales']
        self._smplx_rotation=vertex_gaussian_assets['rotations']
        self._smplx_opacity=vertex_gaussian_assets['opacities']
        self._smplx_xyz=vertex_gaussian_assets['positions']
        self._smplx_offset=vertex_gaussian_assets['static_offsets']

        self._uv_scaling=uv_gaussian_assets['scales']
        self._uv_rotation=uv_gaussian_assets['rotations']
        self._uv_opacity=uv_gaussian_assets['opacities']
        self._uv_local_xyz=uv_gaussian_assets['local_pos']
        self._uv_binding_face=uv_gaussian_assets['binding_face'][0].squeeze(-1) #same for all batch
        self._uv_face_bary=uv_gaussian_assets['face_bary'][0].squeeze(-1) #same for all batch
        
        if self.cfg.with_neural_refiner:
            self._smplx_features_color=vertex_gaussian_assets['colors']
            self._uv_features_color=uv_gaussian_assets['colors']
            self._smplx_features_color[...,:3]=torch.sigmoid(self._smplx_features_color[...,:3]) 
            self._uv_features_color[...,:3]=torch.sigmoid(self._uv_features_color[...,:3])
            self._smplx_features_color[...,3]= self._smplx_features_color[...,3] * 0 + 1
            self._uv_features_color[...,3]=self._uv_features_color[...,3] * 0 + 1
        else:
            self._smplx_features_dc=vertex_gaussian_assets['colors'][:,:,:3]
            self._smplx_features_rest=vertex_gaussian_assets['colors'][:,:,3:]
            self._uv_features_dc=uv_gaussian_assets['colors'][:,:,:3]
            self._uv_features_rest=uv_gaussian_assets['colors'][:,:,3:]
            
            b,n=self._smplx_xyz.shape[0],self._smplx_xyz.shape[1]
            self._smplx_features_dc=self._smplx_features_dc.reshape(b,n,-1,3)
            self._smplx_features_rest=self._smplx_features_rest.reshape(b,n,-1,3)
            b,n=self._uv_local_xyz.shape[0],self._uv_local_xyz.shape[1]
            self._uv_features_dc=self._uv_features_dc.reshape(b,n,-1,3)
            self._uv_features_rest=self._uv_features_rest.reshape(b,n,-1,3)
        
        self.smplx=None
        self._canoical=False
        if pruning:
            self.prune_gaussians()
            
    def init_ehm(self,ehm=None):
        if self.with_smplx_tracking:
            if ehm is None:
                self.smplx=SMPLX(self.cfg.smplx_assets_dir,add_teeth=self.cfg.add_teeth,uv_size=self.cfg.uvmap_size).to(self._smplx_xyz.device)
            else:
                self.smplx=ehm.to(self._smplx_xyz.device)
            self.ehm=self.smplx
        else:
            if ehm is None:
                self.ehm=EHM_v2(self.cfg.flame_assets_dir,self.cfg.smplx_assets_dir,self.cfg.mano_assets_dir,add_teeth=True,uv_size=self.cfg.uvmap_size).to(self._smplx_xyz.device)
            else:
                self.ehm=ehm.to(self._smplx_xyz.device) #.to(self.device)
            self.smplx=self.ehm.smplx

    def prune_gaussians(self):
        #prune gaussians with opacity less than threshold
        assert self._uv_opacity.shape[0]==1 #assert batch size is 1
        mask=self._uv_opacity>self.opacity_threshold
        mask=mask.squeeze(-1)
        mask_bool = mask[0].bool()
        
        self._uv_scaling=(self._uv_scaling[:,mask_bool])
        self._uv_rotation=(self._uv_rotation[:,mask_bool])
        self._uv_opacity=(self._uv_opacity[:,mask_bool])
        self._uv_local_xyz=(self._uv_local_xyz[:,mask_bool])
        self._uv_binding_face=self._uv_binding_face[mask_bool]
        self._uv_face_bary=self._uv_face_bary[mask_bool]
        
        if self.cfg.with_neural_refiner:
            self._uv_features_color=(self._uv_features_color[:,mask_bool])
        else:
            self._uv_features_dc=(self._uv_features_dc[:,mask_bool])
            self._uv_features_rest=(self._uv_features_rest[:,mask_bool])
    
    def forward(self,batch, pd_smplx_dict=None):
        #batch: traget pose expr
        #smplx vertex gaussians
        
        batch_size=self._smplx_xyz.shape[0]
        deformed_assets={}

        if pd_smplx_dict:
            smplx_deform_res = pd_smplx_dict
        else:
            if self.with_smplx_tracking:  # False
                smplx_deform_res=self.smplx(batch['smplx_coeffs'])
            else:
                smplx_deform_res=self.ehm(batch['smplx_coeffs'],batch['flame_coeffs']) #  可以复用 , static_offset=static_offset
                

        if self.with_smplx_gaussian: # True
            self._smplx_xyz_deform = smplx_deform_res["vertices"]  # [B, 10595, 3]
            d_deform_rot_xyzw = rotmat_to_unitquat(smplx_deform_res["ver_transform_mat"][:,:,:3,:3])  # [B, 10595, 4]
            self._smplx_rotation_deform=torch.nn.functional.normalize(quat_xyzw_to_wxyz(quat_product(d_deform_rot_xyzw,quat_wxyz_to_xyzw(self._smplx_rotation))),dim=-1)
        else:
            self._smplx_xyz_deform=self._smplx_xyz
            self._smplx_rotation_deform=self._smplx_rotation
        
        if self.with_uv_gaussian and self.with_smplx_gaussian:
            #uvmap gaussians  给每个三角面片构建一个局部坐标系（三个正交方向向量），也可以估算面片大小。
            # [B,f,3,3] face_orien_mat   face_scaling [B,f,1]
            face_orien_mat, face_scaling = compute_face_orientation(smplx_deform_res["vertices"], self.smplx.faces_tensor, return_scale=True)
            face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat)) # [B,f,4]
            # n 是 uv map 的像素数目（19w），k 是每个 uv pixel 对应的面片顶点数目（通常为3）
            face_vertices = smplx_deform_res["vertices"][:,self.smplx.faces_tensor] # b f k 3  # 每个face 3个顶点，k=3
            face_vertices_nn = face_vertices[:,self._uv_binding_face] # b n k 3 每个 uv pixel 对应的face的3个顶点
            face_bary = self._uv_face_bary[None].expand(batch_size,-1,-1) # b n k
            face_center_nn = torch.einsum('bnk,bnkj->bnj',face_bary, face_vertices_nn) # B n 3  每个uv pixel对应的 face 的中心坐标
            
            # or
            # face_center = smplx_deform_res["vertices"][:,self.smplx.faces_tensor].mean(-2)
            # face_center_nn=face_center[:,self._uv_binding_face]
            
            face_scaling_nn = face_scaling[:,self._uv_binding_face]
            
            xyz=torch.einsum('bnij,bnj->bni',face_orien_mat[:,self._uv_binding_face], self._uv_local_xyz)
            self._uv_xyz_deform = face_center_nn + xyz * face_scaling_nn     #  面片旋转后的局部坐标 * 缩放 + 3D 面片中心 
            
            face_orien_quat=face_orien_quat[:,self._uv_binding_face] # torch.nn.functional.normalize(,dim=-1)
            self._uv_rotation_deform=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(self._uv_rotation)))
            self._uv_scaling_deform = self._uv_scaling * face_scaling_nn
            
            # integrate
            self._xyz_deform=  torch.cat([self._smplx_xyz_deform, self._uv_xyz_deform],dim=1) # [B,10595 + 199141,3] 
            self._rotation_deform=    torch.cat([self._smplx_rotation_deform, self._uv_rotation_deform],dim=1)
            self._scaling_deform= torch.cat([self._smplx_scaling,self._uv_scaling_deform],dim=1)
            self._opacity_deform= torch.cat([self._smplx_opacity,self._uv_opacity],dim=1)
            
            if self.cfg.with_neural_refiner:
                self._features_color = torch.cat([self._smplx_features_color,self._uv_features_color],dim=1) # [B, 10595 + 199141, 32]
                deformed_assets.update({'features_color':self._features_color})
            else:
                self._features_dc=torch.cat([self._smplx_features_dc,self._uv_features_dc],dim=1)
                self._features_rest=torch.cat([self._smplx_features_rest,self._uv_features_rest],dim=1)
                shs=torch.cat([self._features_dc,self._features_rest],dim=2)
                deformed_assets.update({'shs': shs,})
                
        elif  self.with_uv_gaussian and not self.with_smplx_gaussian:
            #uvmap gaussians
            face_orien_mat, face_scaling = compute_face_orientation(smplx_deform_res["vertices"], self.smplx.faces_tensor, return_scale=True)
            face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat))
            
            face_vertices=smplx_deform_res["vertices"][:,self.smplx.faces_tensor]# b f k 3
            face_vertices_nn=face_vertices[:,self._uv_binding_face]# b n k 3
            face_bary=self._uv_face_bary[None].expand(batch_size,-1,-1)# b n k
            face_center_nn= torch.einsum('bnk,bnkj->bnj',face_bary,face_vertices_nn)# B n 3

            face_scaling_nn=face_scaling[:,self._uv_binding_face]
            
            xyz=torch.einsum('bnij,bnj->bni',face_orien_mat[:,self._uv_binding_face], self._uv_local_xyz)
            self._uv_xyz_deform=xyz * face_scaling_nn + face_center_nn
            
            face_orien_quat=face_orien_quat[:,self._uv_binding_face]#torch.nn.functional.normalize(,dim=-1)
            self._uv_rotation_deform=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(self._uv_rotation)))
            self._uv_scaling_deform = self._uv_scaling* face_scaling_nn
            
            self._xyz_deform=self._uv_xyz_deform
            self._rotation_deform=self._uv_rotation_deform
            self._scaling_deform=self._uv_scaling_deform
            self._opacity_deform=self._uv_opacity
            assert self.cfg.with_neural_refiner
            deformed_assets.update({'features_color':self._uv_features_color})
            
        else:
            self._xyz_deform=self._smplx_xyz_deform
            self._rotation_deform=self._smplx_rotation_deform
            self._scaling_deform=self._smplx_scaling
            self._opacity_deform=self._smplx_opacity
            assert self.cfg.with_neural_refiner
            deformed_assets.update({'features_color':self._smplx_features_color})

            
        deformed_assets.update( {
        'xyz': self._xyz_deform,
        'rotation': self._rotation_deform, 
        'scaling': self._scaling_deform, 
        'opacity': self._opacity_deform, 
        'sh_degree':self.max_sh_degree,
        'smplx_xyz_deform':smplx_deform_res["vertices"],
        })
        

        
        return deformed_assets

    def get_canoical_gaussians(self):
        if not self.with_smplx_gaussian or not self.with_uv_gaussian or not self.with_neural_refiner: return
        #uvmap gaussians
        batch_size=self._smplx_xyz.shape[0]
        v_template=self._smplx_xyz.clone().expand(batch_size,-1,-1)
        if self._smplx_offset is not None:
            v_template=v_template+self._smplx_offset
        face_orien_mat, face_scaling = compute_face_orientation(v_template, self.smplx.faces_tensor, return_scale=True)
        face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat))
        
        face_vertices=v_template[:,self.smplx.faces_tensor] # b f k 3
        face_vertices_nn=face_vertices[:,self._uv_binding_face] # b n k 3
        face_bary=self._uv_face_bary[None].expand(batch_size,-1,-1) # b n k
        face_center_nn= torch.einsum('bnk,bnkj->bnj',face_bary,face_vertices_nn)# B n 3
        #or
        # face_center=v_template[:,self.smplx.faces_tensor].mean(-2)
        # face_center_nn=face_center[:,self._uv_binding_face]
        
        face_scaling_nn=face_scaling[:,self._uv_binding_face]
        
        xyz=torch.einsum('bnij,bnj->bni',face_orien_mat[:,self._uv_binding_face], self._uv_local_xyz)
        self._uv_xyz_cano=xyz * face_scaling_nn + face_center_nn
        
        face_orien_quat=torch.nn.functional.normalize(face_orien_quat[:,self._uv_binding_face],dim=-1)
        self._uv_rotation_cano=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(self._uv_rotation)))
        self._uv_scaling_cano = self._uv_scaling* face_scaling_nn
        self._uv_opacity_cano=self._uv_opacity
        self._canoical=True

    def save_point_ply(self,save_path,save_split=True,assets=None):
        if not self.with_smplx_gaussian or  not self.with_neural_refiner or not self.with_uv_gaussian: return
        if not self._canoical:
            self.get_canoical_gaussians()
        
        if assets is not None:
            xyz_all_np=assets['xyz'][0].detach().cpu().numpy()
            colors_all_np=assets['features_color'][0,...,:3].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_all_np)
            pcd.colors = o3d.utility.Vector3dVector(colors_all_np)
            o3d.io.write_point_cloud(os.path.join(save_path,'deformed.ply'), pcd)
        else:
            xyz_all_np=torch.cat([self._smplx_xyz,self._uv_xyz_cano],dim=1).detach().cpu().numpy()
            colors_all_np=torch.cat([self._smplx_features_color[...,:3],self._uv_features_color[...,:3]],dim=1).detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_all_np[0])
            pcd.colors = o3d.utility.Vector3dVector(colors_all_np[0])
            o3d.io.write_point_cloud(os.path.join(save_path,'canonical.ply'), pcd)
            
            if save_split:
                xyz_smplx_np=self._smplx_xyz.detach().cpu().numpy()
                colors_smplx_np=self._smplx_features_color[...,:3].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_smplx_np[0])
                pcd.colors = o3d.utility.Vector3dVector(colors_smplx_np[0])
                o3d.io.write_point_cloud(os.path.join(save_path,'canonical_smplx.ply'), pcd)

                xyz_uv_np=self._uv_xyz_cano.detach().cpu().numpy()
                colors_uv_np=self._uv_features_color[...,:3].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_uv_np[0])
                pcd.colors = o3d.utility.Vector3dVector(colors_uv_np[0])
                o3d.io.write_point_cloud(os.path.join(save_path,'canonical_uv.ply'), pcd)
    

        
    def save_gaussian_ply(self,save_path,save_split=False):
        if not self.with_smplx_gaussian or not self.with_neural_refiner or not self.with_uv_gaussian: return
        if not self._canoical:
            self.get_canoical_gaussians()
            
        xyz_all_np=torch.cat([self._smplx_xyz,self._uv_xyz_cano],dim=1)[0].detach().cpu().numpy()
        colors_all_np=torch.cat([self._smplx_features_color[0,:,:3],self._uv_features_color[0,:,:3]],dim=0)[:,None]
        colors_all_np=(colors_all_np) / 0.28209479177387814 #RGB to SHS
        opacities_all_np = torch.cat([inverse_sigmoid(self._smplx_opacity),
                                      inverse_sigmoid(self._uv_opacity_cano)],dim=1)[0].detach().cpu().numpy()
        scale_all_np = torch.cat([torch.log(self._smplx_scaling),
                                  torch.log(self._uv_scaling_cano)],dim=1)[0].detach().cpu().numpy()
        rotation_all_np = torch.cat([self._smplx_rotation,self._uv_rotation_cano],dim=1)[0].detach().cpu().numpy()
        
        normals = np.zeros_like(xyz_all_np)
        f_dc = colors_all_np.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest=torch.zeros((f_dc.shape[0],0,3))
        f_rest=f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz_all_np.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz_all_np, normals, f_dc, f_rest, opacities_all_np, scale_all_np, rotation_all_np), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(save_path, 'GS_canonical.ply'))
        
        if save_split:
            xyz_smplx_np=self._smplx_xyz[0].detach().cpu().numpy()
            colors_smplx_np=self._smplx_features_color[0,...,:3][:,None]
            colors_smplx_np=(colors_smplx_np ) / 0.28209479177387814 #RGB to SHS
            opacities_smplx_np = inverse_sigmoid(self._smplx_opacity[0]).detach().cpu().numpy()
            scale_smplx_np = torch.log(self._smplx_scaling[0]).detach().cpu().numpy()
            rotation_smplx_np = self._smplx_rotation[0].detach().cpu().numpy()
            normals = np.zeros_like(xyz_smplx_np)
            f_dc = colors_smplx_np.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest=torch.empty((f_dc.shape[0],0,3))
            f_rest=f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate((xyz_smplx_np, normals, f_dc, f_rest, opacities_smplx_np, scale_smplx_np, rotation_smplx_np), axis=1)
            elements = np.empty(xyz_smplx_np.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(os.path.join(save_path, 'GS_canonical_smplx.ply'))

            xyz_uv_np=self._uv_xyz_cano[0].detach().cpu().numpy()
            colors_uv_np=self._uv_features_color[0,...,:3][:,None]
            colors_uv_np=(colors_uv_np ) / 0.28209479177387814 #RGB to SHS
            opacities_uv_np = inverse_sigmoid(self._uv_opacity_cano[0]).detach().cpu().numpy()
            scale_uv_np = torch.log(self._uv_scaling_cano[0]).detach().cpu().numpy()
            rotation_uv_np = self._uv_rotation_cano[0].detach().cpu().numpy()
            normals = np.zeros_like(xyz_uv_np)
            f_dc = colors_uv_np.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest=torch.empty((f_dc.shape[0],0,3))
            f_rest=f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate((xyz_uv_np, normals, f_dc, f_rest, opacities_uv_np, scale_uv_np, rotation_uv_np), axis=1)
            elements = np.empty(xyz_uv_np.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(os.path.join(save_path, 'GS_canonical_uv.ply'))
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(0):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

# def configure_optimizers(ehm_model):

#     params_main = filter(lambda p: p.requires_grad, ehm_model._params_main())
#     optimizer_main = torch.optim.AdamW( lr = 1e-05,  weight_decay =0.0001, params=params_main)
#     return optimizer_main


def configure_render_optimizers(infer_model,cfg,render_model=None):
    learning_rate = cfg.learning_rate
    print('Learning rate: {}'.format(learning_rate))
    # params
    decay_names = []
    normal_params, decay_params0, decay_params1 = [], [], []

    def process_model_parameters(model):
        nonlocal decay_names, normal_params, decay_params0, decay_params1
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'style_mlp' in name or 'final_linear' in name:
                decay_names.append(".".join(name.split('.')[:-2]) if len(name.split('.')) > 3 else ".".join(name.split('.')[:-1]))
                decay_params0.append(param)

            elif 'up_point_decoder' in name or ('vertex_gs_decoder' in name and 'feature_layers' not in name) \
                  or 'uv_feature_decoder' in name or 'prj_feature_decoder' in name:
                decay_params1.append(param)
            else:
                normal_params.append(param)

    process_model_parameters(infer_model)
    if render_model is not None:
        process_model_parameters(render_model)

    # optimizer
    # learning_rate
    learning_rate = learning_rate * 0.05  # 学习率都变小为 0.2 倍
    optimizer = torch.optim.Adam([
            {'params': normal_params, 'lr': learning_rate}, 
            {'params': decay_params0, 'lr': learning_rate*0.1},
            {'params': decay_params1, 'lr': learning_rate},
        ], lr=learning_rate, betas=(0.0, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor= 0.1,   #   cfg.lr_decay_rate,   # end_factor = 0.03
        total_iters=cfg.lr_decay_iter,  # 20 w
    )
    return optimizer, scheduler

def configure_ehm_optimizers(ehm_model, cfg, render_model=None):
    learning_rate = cfg.learning_rate
    print('Learning rate: {}'.format(learning_rate))

    def params_main(ehm_model):
        return list(ehm_model.head.parameters()) + list(ehm_model.backbone.parameters())


    ehm_params = filter(lambda p: p.requires_grad, params_main(ehm_model))
    # optimizer
    # learning_rate
    learning_rate = learning_rate * 0.05  # 学习率都变小为 0.2 倍
    optimizer = torch.optim.Adam([
            {'params': ehm_params, 'lr': 1e-06},  # 学习率再小一些, 原本是 1e-05
        ], lr=learning_rate, betas=(0.0, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=  0.1 , # cfg.lr_decay_rate,   # end_factor = 0.03
        total_iters=cfg.lr_decay_iter,  # 20 w
    )
    return optimizer, scheduler

def linear_interpolate_transform_mats(transfor_mats,W):
    #direct linear interpolate transform mats will cause the rotation matrix not orthogonal
    #torch.matmul(up_lbs_weights, transform_mats.view(batch_size, up_lbs_weights.shape[-1], 16)).view(batch_size, -1, 4, 4)

    batch_size,n=W.shape[0],W.shape[1]
    transform_quad=rotmat_to_unitquat(transfor_mats[:,:,:3,:3])
    linear_transform_quad=torch.einsum('bnj,bjq->bnq',W,transform_quad)# b n 4
    linear_transform_quad=torch.nn.functional.normalize(linear_transform_quad,dim=-1)
    linear_transform_mat=unitquat_to_rotmat(linear_transform_quad)# b n 3 3
    linear_T=torch.einsum('bnj,bjq->bnq',W,transfor_mats[:,:,:3,3])# b n 3
    resutl_transform=torch.zeros((batch_size,n,4,4),device=W.device,dtype=torch.float32)
    resutl_transform[:,:,:3,:3]=linear_transform_mat
    resutl_transform[:,:,:3,3]=linear_T
    resutl_transform[:,:,3,3]=1.0
    
    return resutl_transform

def get_persudo_view_dirs(xy_image_coord,c2w_cam,focal,):
    # get view dirs(cam origin to points (assume zero depth offset)) in wold coordinate
    # with each camera with same focal
    batch_size=c2w_cam.shape[0]
    xy_cam_coord=xy_image_coord*focal
    z_cam_coord=torch.ones_like(xy_cam_coord[:,:,None,0])
    xyz_cam_coord=torch.cat([xy_cam_coord,z_cam_coord],dim=-1)
    xyz_cam_coord=xyz_cam_coord[None].expand(batch_size,-1,-1,-1)
    xyz_world_coord=torch.einsum('bij,bhwj->bhwi', c2w_cam[:, :3, :3], xyz_cam_coord)
    view_dirs = nn.functional.normalize(xyz_world_coord, p=2, dim=-1)
    return view_dirs

def get_smplx_vertex_view_dirs(deform_vertices,c2w_cam):
    # get view dirs(cam origin to smplx vertices in wold coordinate
    batch_size=c2w_cam.shape[0]
    cam_origin=c2w_cam[:,:3,3]
    deform_vertices=deform_vertices[None].expand(batch_size,-1,-1)
    view_dirs=(deform_vertices-cam_origin[:,None,:])
    view_dirs=nn.functional.normalize(view_dirs,p=2,dim=-1)
    
    return view_dirs

def get_cam_dirs(_cam):
    # get the z axis of cam in wold coordinate
    batch_size=_cam.shape[0]
    z_dirs=torch.tensor([[0,0,1]],dtype=torch.float32,device=_cam.device)
    z_dirs=z_dirs.expand(batch_size,-1)
    z_dirs=torch.einsum('bij,bj->bi', _cam[:, :3, :3], z_dirs)#b 3
    return z_dirs

def get_pixel_coordinates(image_height, image_width):
    x_range = torch.arange(0, image_width, dtype=torch.float32)+0.5
    y_range = torch.arange(0, image_height, dtype=torch.float32)+0.5

    coords = torch.cartesian_prod(x_range, y_range)

    coords = coords[:, [1, 0]]
    coords=coords.reshape(image_height,image_width,-1)
    return coords

def save_visual_ply(up_point_gs_dict,ehm_deform_res,images=None):
    import open3d as o3d
    batch_size=up_point_gs_dict['positions'].shape[0]
    uppos=up_point_gs_dict['positions']
    if len(up_point_gs_dict['positions'].shape)==4:
        uppos=uppos.reshape(batch_size,-1,3)
    all_points=torch.cat([uppos,ehm_deform_res['vertices']],dim=1)
    
    if images is not None: img_colors=torch.nn.functional.interpolate(images,(296,296)).permute(0,2,3,1).reshape(batch_size,-1,3)
    all_points_np=all_points[0].detach().cpu().numpy()
    all_points_colors=(all_points_np-all_points_np.min(axis=0))/(all_points_np.max(axis=0)-all_points_np.min(axis=0))
    all_points_colors[:296**2,:3]=img_colors[0].detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_np)
    pcd.colors = o3d.utility.Vector3dVector(all_points_colors)
    o3d.io.write_point_cloud("debug_output.ply", pcd)

def save_visual_prj(vertex_prj,batch_images):
    import torchvision.utils as vutils
    vertex_prj=vertex_prj[0,0]
    batch_image=batch_images[0]
    canvas_size=batch_image.shape[1]
    vertex_prj=vertex_prj*canvas_size/2+canvas_size/2
    vertex_prj = torch.clamp(vertex_prj, 0, canvas_size - 1)
    vertex_prj = vertex_prj.int()
    canvas=batch_image
    for point in vertex_prj:
        x, y = point[0], point[1]
        canvas[0, y, x] = 1.0
    output_path = "vertex_prj_visualization.jpg"
    vutils.save_image(canvas.float(), output_path)
    
    #cameras.get_ndc_to_screen_transform(cameras, with_xyflip=True, image_size=image_size)


def save_visual_images(images,save_path):
    import torchvision.utils as vutils
    vutils.save_image(images, save_path)

def save_tensor_as_heatmap(
    tensor: torch.Tensor,
    save_path: str,
    cmap: str = 'turbo', #'hot', 'plasma', 'inferno', 'magma', 'cividis' viridis
    vmin: float = 0.0,
    vmax: float = 1.0,
    dpi: int = 100
):
    import matplotlib.pyplot as plt
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0) 
    arr = tensor.detach().cpu().numpy()
    

    plt.figure(figsize=(arr.shape[1]/dpi, arr.shape[0]/dpi), dpi=dpi)
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()