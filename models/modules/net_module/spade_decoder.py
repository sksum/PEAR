import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class SPADEDecoder(nn.Module):
    def __init__(self,in_dim=32,mid_dim=8,label_dim=16,num_mid_blocks=4,num_up_blocks=0):
        super().__init__()
        ic = in_dim
        oc = mid_dim
        norm_G = 'spadespectralinstance'
        label_nc = label_dim
        self.num_up_blocks=num_up_blocks
        self.num_mid_blocks=num_mid_blocks
        
        self.fc = nn.Conv2d(ic, oc, 3, padding=1)
        self.f_label = nn.Conv2d(ic, label_nc, 3, padding=1)
        self.G_middle_blocks=nn.ModuleList()
        
        for i in range(num_mid_blocks):
            self.G_middle_blocks.append(SPADEResnetBlock(oc, oc, norm_G, label_nc))

        if num_up_blocks>0:
            self.G_up_blocks=nn.ModuleList()
            for i in range(num_up_blocks):
                 self.G_up_blocks.append(nn.Sequential(nn.Upsample(scale_factor=2),SPADEResnetBlock(oc, oc, norm_G, label_nc)))
            
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        # self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        #seg=feature
        seg = self.f_label(feature)
        x = self.fc(feature)

        for i in range(self.num_mid_blocks):
            x=self.G_middle_blocks[i](x,seg)
        for i in range(self.num_up_blocks):
            x=self.G_up_blocks[i](x,seg)
            
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.sigmoid(x)
        return x
    

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')#nearest
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

if __name__ == '__main__':
    import time
    from tqdm import tqdm
    with torch.no_grad():
        decode_channel=32
        model = SPADEDecoder(in_dim=decode_channel,mid_dim=16,label_dim=16,num_mid_blocks=1,num_up_blocks=0).cuda()
        model.eval()
        
        data = torch.rand(1, decode_channel, 512, 512).cuda()
        print(model(data.clone()).shape)
        iterations = 100 
        start_time = time.time()
        for _ in tqdm(range(iterations)):
            result = model(data.clone())
        total_time = time.time() - start_time
        average_time = total_time / iterations
        fps = 1 / average_time
        print(f"Decode_channel:{decode_channel}. FPS (Frames Per Second): {fps:.2f}")
        # import pdb; pdb.set_trace()