import torch
import torch.nn as nn
import lightning as L
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_32 import GaussianRasterizationSettings, GaussianRasterizer_32
from models.modules.net_module import Nueral_Refiner_Model
import time
class GaussianRenderer(L.LightningModule): 

    def __init__(self,cfg):
        super(GaussianRenderer, self).__init__()
        self.cfg=cfg
        if self.cfg.with_neural_refiner:
            refiner_type =self.cfg.refiner_type
            refiner_config = self.cfg[refiner_type]
            self.nerual_refiner = Nueral_Refiner_Model[refiner_type](**refiner_config)
    
    def forward(self,*args, **kwargs):
        # if self.cfg.with_neural_refiner:
        #     return self.forward_nueral_refine(*args, **kwargs)
        # else:
        #     return self.forward_shs(*args, **kwargs)
        return self.forward_nueral_refine(*args, **kwargs)
    
    def forward_shs(self, gaussian_assets, cam_params, bg=None,scaling_modifier=1.0,antialiasing=False):
        
        mean_3d = gaussian_assets['xyz']
        opacity = gaussian_assets['opacity']
        scales = gaussian_assets['scaling']
        rotations = gaussian_assets['rotation']
        shs = gaussian_assets['shs']
        sh_degree = gaussian_assets['sh_degree']
        batch_size = mean_3d.shape[0]
        device=mean_3d.device
        
        bg=torch.ones((batch_size,3),dtype=torch.float32,device=device)*bg
            
        mean_2d = torch.zeros_like(mean_3d, dtype=torch.float32, requires_grad=True, device=device)
        mean_2d.retain_grad()
        cov3D_precomp = None
        
        rendered_images,radiis,depth_images=[],[],[]
        
        for bi in range(batch_size):
            raster_settings = GaussianRasterizationSettings(
                image_height=int(cam_params['image_height'][bi]),
                image_width=int(cam_params['image_width'][bi]),
                tanfovx=float(cam_params['tanfovx'][bi]),
                tanfovy=float(cam_params['tanfovy'][bi]),
                bg=bg[bi],
                scale_modifier=scaling_modifier,
                viewmatrix=cam_params['world_view_transform'][bi],
                projmatrix=cam_params['full_proj_transform'][bi],
                sh_degree=int(sh_degree),
                campos=cam_params['camera_center'][bi],
                prefiltered=False,
                debug=False,
                antialiasing=antialiasing
                )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            if sh_degree>0:
                rendered_image, radii, depth_image = rasterizer(
                        means3D = mean_3d[bi],
                        means2D = mean_2d[bi],
                        shs = shs[bi],
                        colors_precomp = None,
                        opacities = opacity[bi],
                        scales = scales[bi],
                        rotations = rotations[bi],
                        cov3D_precomp = cov3D_precomp)
            else:
                rendered_image, radii, depth_image = rasterizer(
                        means3D = mean_3d[bi],
                        means2D = mean_2d[bi],
                        shs = None,
                        colors_precomp = shs[bi,:,0],
                        opacities = opacity[bi],
                        scales = scales[bi],
                        rotations = rotations[bi],
                        cov3D_precomp = cov3D_precomp)
            rendered_image = rendered_image.clamp(0, 1)
            
            rendered_images.append(rendered_image)
            radiis.append(radii)
            depth_images.append(depth_image)
        
        rendered_images = torch.stack(rendered_images, dim=0)
        radiis = torch.stack(radiis, dim=0)
        depth_images = torch.stack(depth_images, dim=0)
        
        return {
            "renders": rendered_images, 
            "viewspace_points": mean_2d,
            "radiis": radiis,
            "depths" : depth_images,
            # "visibility_filter" : (radiis > 0).nonzero(),
            }
        
    def forward_nueral_refine(self, gaussian_assets, cam_params, bg=None,scaling_modifier=1.0,antialiasing=False):
        
        mean_3d = gaussian_assets['xyz']
        opacity = gaussian_assets['opacity']
        scales = gaussian_assets['scaling']
        rotations = gaussian_assets['rotation']
        features_color = gaussian_assets['features_color']
        sh_degree = 0
        batch_size = mean_3d.shape[0]
        device=mean_3d.device
        
        bg=torch.ones((batch_size,features_color.shape[-1]),dtype=torch.float32,device=device)*bg
        # features_color[:,3:4] = features_color[:,3:4] * 0 + 1.0   # 
            
        mean_2d = torch.zeros_like(mean_3d, dtype=torch.float32, requires_grad=True, device=device)
        mean_2d.retain_grad()
        cov3D_precomp = None
        
        rendered_images,radiis,depth_images=[],[],[]

        for bi in range(batch_size):
            raster_settings = GaussianRasterizationSettings(
                image_height=int(cam_params['image_height'][bi]),
                image_width=int(cam_params['image_width'][bi]),
                tanfovx=float(cam_params['tanfovx'][bi]),
                tanfovy=float(cam_params['tanfovy'][bi]),
                bg=bg[bi],
                scale_modifier=scaling_modifier,
                viewmatrix=cam_params['world_view_transform'][bi],
                projmatrix=cam_params['full_proj_transform'][bi],
                sh_degree=int(sh_degree),
                campos=cam_params['camera_center'][bi],
                prefiltered=False,
                debug=False,
                antialiasing=antialiasing
                )
            rasterizer = GaussianRasterizer_32(raster_settings=raster_settings)
            rendered_image, radii, depth_image = rasterizer(
                    means3D = mean_3d[bi],
                    means2D = mean_2d[bi],
                    shs = None,
                    colors_precomp = features_color[bi],  # 这是一个 32 channel 的 feature
                    opacities = opacity[bi],
                    scales = scales[bi],
                    rotations = rotations[bi],
                    cov3D_precomp = cov3D_precomp)
        
            rendered_images.append(rendered_image)
            radiis.append(radii)
            depth_images.append(depth_image)
        
        
        rendered_images = torch.stack(rendered_images, dim=0) 
        raw_images = rendered_images[:,:3] 
        render_masks = rendered_images[:,3:4] 
        # rendered_images = rendered_images[:,:3] 
        # if self.cfg.with_neural_refiner:  
        #     refine_images = self.nerual_refiner(rendered_images)   
        # else:
        refine_images =   self.nerual_refiner(rendered_images)    # rendered_images[:,:3]

        radiis = torch.stack(radiis, dim=0)
        depth_images = torch.stack(depth_images, dim=0)

        return {
            "renders": refine_images ,  # rendered_images,  # refine_images,
            "raw_renders": raw_images,  
            "viewspace_points": mean_2d,
            "radiis": radiis,
            "depths" : depth_images,
            'extra_renders':rendered_images[:,3:4],
            'render_masks': render_masks, 
            # "visibility_filter" : (radiis > 0).nonzero(),
            }

    def forward_drawing_figs(self, gaussian_assets, cam_params, bg=None,scaling_modifier=1.0,antialiasing=False):
        mean_3d = gaussian_assets['xyz']
        opacity = gaussian_assets['opacity']
        scales = gaussian_assets['scaling']
        rotations = gaussian_assets['rotation']
        features_color = gaussian_assets['features_color']
        sh_degree = 0
        batch_size = mean_3d.shape[0]
        device=mean_3d.device
        
        bg=torch.ones((batch_size,features_color.shape[-1]),dtype=torch.float32,device=device)*bg
            
        mean_2d = torch.zeros_like(mean_3d, dtype=torch.float32, requires_grad=True, device=device)
        mean_2d.retain_grad()
        cov3D_precomp = None
        
        rendered_images,radiis,depth_images=[],[],[]

        for bi in range(batch_size):
            raster_settings = GaussianRasterizationSettings(
                image_height=int(cam_params['image_height'][bi]),
                image_width=int(cam_params['image_width'][bi]),
                tanfovx=float(cam_params['tanfovx'][bi]),
                tanfovy=float(cam_params['tanfovy'][bi]),
                bg=bg[bi],
                scale_modifier=scaling_modifier,
                viewmatrix=cam_params['world_view_transform'][bi],
                projmatrix=cam_params['full_proj_transform'][bi],
                sh_degree=int(sh_degree),
                campos=cam_params['camera_center'][bi],
                prefiltered=False,
                debug=False,
                antialiasing=antialiasing
                )
            rasterizer = GaussianRasterizer_32(raster_settings=raster_settings)
            rendered_image, radii, depth_image = rasterizer(
                    means3D = mean_3d[bi],
                    means2D = mean_2d[bi],
                    shs = None,
                    colors_precomp = features_color[bi],
                    opacities = opacity[bi],
                    scales = scales[bi],
                    rotations = rotations[bi],
                    cov3D_precomp = cov3D_precomp)
            
            rendered_images.append(rendered_image)
            radiis.append(radii)
            depth_images.append(depth_image)
        
        
        rendered_images = torch.stack(rendered_images, dim=0)
        raw_images = rendered_images[:,:3]
    
        radiis = torch.stack(radiis, dim=0)
        depth_images = torch.stack(depth_images, dim=0)
        
        return {
            "raw_renders": raw_images,  
            "viewspace_points": mean_2d,
            "radiis": radiis,
            "depths" : depth_images,
            'extra_renders':rendered_images[:,3:4],
            # "visibility_filter" : (radiis > 0).nonzero(),
            }