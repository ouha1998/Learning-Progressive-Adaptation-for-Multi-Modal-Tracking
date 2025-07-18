import torch
from torch import nn
import timm
import math
from .utils import token2feature

class Adapterwozeros(nn.Module):
    # def __init__(self, dim=8, xavier_init=False):
    def __init__(self, dim=192):
        super().__init__()

        self.dropout = nn.Dropout(0.1)

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)

    def forward(self, x):

        B, N, C = x.shape
        x_down = self.adapter_down(x)
        # x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        # x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        # x = x.view(B,16,16,-1).permute(0,3,1,2)
        # x_down = self.adapter_down(x)
        # x_down = x_down.permute(0,2,3,1).view(B,N,C)
        # #x_down = self.act(x_down)
        # x_down = self.adapter_mid(x_down)
        # x_down = x_down.view(B,16,16,-1).permute(0,3,1,2)
        # #x_down = self.act(x_down)
        # x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)
        # x_up = x_up.permute(0,2,3,1).view(B,N,C)

        return x_up

class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2 #192
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2 #384
        self.pool_dim = pool_dim = pool_in * 2 #672

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)#192,384
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim) #384,384
        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)#3,1,1
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0) #672,672
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W   32,384,8,8
        #DWCONV
        cx = x[:, :self.cnn_in, :, :].contiguous()#48,8,8
        cx = self.conv1(cx)#32,96,8,8
        cx = self.proj1(cx)#32,96,8,8
        cx = self.mid_gelu1(cx)#32,96,8,8
        #MAXPOOL
        px = x[:, self.cnn_in:, :, :].contiguous()#2,8,8
        px = self.Maxpool(px)#32,2,8,8
        px = self.proj2(px)#32,4,8,8
        px = self.mid_gelu2(px)#32,96,8,8

        hx = torch.cat((cx, px), dim=1) #32,8,8,8
        return hx


class LowMixer(nn.Module):
    def __init__(self, dim, pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = 8
        self.head_dim = head_dim = dim // 8
        self.dim = dim

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, #2,2
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()#2

    def forward(self, x):
        # B, C, H, W  32,384,8,8
        B, _, _, _ = x.shape
        #B,C,N = x.shape
        xa = self.pool(x)#32,384,4,4
        xa = self.uppool(xa)#32,384,8,8
        return xa


class Mixer(nn.Module):
    def __init__(self, dim=768, proj_drop=0., pool_size=2, act_layer=nn.GELU,smooth=False,
                 **kwargs, ):
        super().__init__()
        #hide = 96
        hide = 8
        # hide = 192
        self.D_fc1 = nn.Linear(768, hide)

        self.low_dim = hide // 2
        self.high_dim = hide // 2
        self.high_mixer = HighMixer(self.high_dim)
        self.low_mixer = LowMixer(self.low_dim,
                                  pool_size=pool_size, )
        self.high_mixer = HighMixer(self.high_dim)
        self.conv_fuse = nn.Conv2d(self.low_dim + self.high_dim * 2, self.low_dim + self.high_dim * 2, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=self.low_dim + self.high_dim * 2)
        self.proj = nn.Conv2d(self.low_dim + self.high_dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = act_layer()
        # self.fovea = Fovea(smooth=smooth)

    def forward(self, x):
        B, L, C = x.shape
        lens_z = 64
        x_z = x[:, 0:lens_z, :]  # 32,64,768
        x_x = x[:, lens_z:, :]
        x_z = token2feature(x_z).permute(0, 2, 3, 1)  # 32,8,8,768
        x_x = token2feature(x_x).permute(0, 2, 3, 1)
        x_z = self.D_fc1(x_z).permute(0,3,1,2)#32,192,8,8
        x_x = self.D_fc1(x_x).permute(0,3,1,2)#32,192,16,16

        # rgb的z高低频
        x_z_h = x_z[:, :self.high_dim, :, :].contiguous()  # 32,4,8,8
        x_z_h = self.high_mixer(x_z_h)  # 32,8,8,8
        x_z_l = x_z[:, self.high_dim:, :, :].contiguous()  # 32,4,8,8
        x_z_l = self.low_mixer(x_z_l)  # 32,4,8,8

        # rgb的x高低频
        x_x_h = x_x[:, :self.high_dim, :, :].contiguous()  # 4，16，16
        x_x_h = self.high_mixer(x_x_h)  # 32,8,16,16
        x_x_l = x_x[:, self.high_dim:, :, :].contiguous()  # 4，16，16
        x_x_l = self.low_mixer(x_x_l)  # 32,4,16,16

        #z
        rgbt_z = torch.cat([x_z_h, x_z_l], dim=1)  # 32,12,8,8
        rgbt_z = rgbt_z + self.conv_fuse(rgbt_z)  # 12,8,8
        rgbt_z = self.proj(rgbt_z)  # 32,768,8,8
        rgbt_z = self.proj_drop(rgbt_z).permute(0, 2, 3, 1)  # 32,8,8,768
        # x
        rgbt_x = torch.cat([x_x_h, x_x_l], dim=1)
        rgbt_x = rgbt_x + self.conv_fuse(rgbt_x)
        rgbt_x = self.proj(rgbt_x)
        rgbt_x = self.proj_drop(rgbt_x).permute(0, 2, 3, 1)

        #rgbt的z,x cat
        x_z = rgbt_z.view(B, -1, 768)  # 64,768
        x_x = rgbt_x.view(B, -1, 768)  # 256,768
        x = torch.cat([x_z, x_x], dim=1)  # 320,768
        return x




'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_direct_adapter_woinit(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.adapter_mid = nn.Linear(dim, dim)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up

"""


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        #print(x.shape)
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        #print(x_down.shape)

        x_patch = x_down[:, 64:].reshape(B, 16, 16, self.dim).permute(0, 3, 1, 2)   ############
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 16 * 16, self.dim)


        #x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up
"""