import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Mixer

class PSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):
        super(PSFM, self).__init__()
        # self.RGBobj = DenseLayer(in_C, out_C)
        # self.Infobj = DenseLayer(in_C, out_C)
        self.obj_fuse = GEFM(cat_C, out_C)
        # self.down = nn.Conv2d(768, in_C, kernel_size=1, stride=1, padding=0, bias=False)
        self.up = nn.Conv2d(out_C,768, kernel_size=1, stride=1, padding=0, bias=False)
        self.down = nn.Conv2d(768,out_C, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, rgb, depth,layer):
        # rgb = self.down(rgb)
        # depth = self.down(depth)
        rgb_sum = self.down(rgb)#B,8,w,h
        Inf_sum = self.down(depth)

        rgb_sum,Inf_sum = self.obj_fuse(rgb_sum, Inf_sum,layer)#B,C,H,W
        rgb_sum = self.up(rgb_sum)
        Inf_sum = self.up(Inf_sum)
        return rgb_sum,Inf_sum


class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.linear1 = nn.Conv2d(out_C, out_C, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear2 = nn.Conv2d(out_C, out_C, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear3 = nn.Conv2d(out_C, out_C, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear4 = nn.Conv2d(out_C, out_C, kernel_size=1, stride=1, padding=0, bias=False)
        self.linearcat = nn.Conv2d(out_C*2, out_C, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x, y,layer):
        Q = self.linearcat(torch.cat([x, y], dim=1))#768,16,16
        RGB_K = self.linear1(x)
        RGB_V = self.linear2(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)#C,L
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)#L,C
        RGB_Q = Q.view(m_batchsize, -1, width * height)#C,L
        RGB_mask = torch.bmm(RGB_K, RGB_Q)#b,256,256

        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        # RGB_refine = self.gamma1 * RGB_refine + y#送入RGB分支
        RGB_refine = RGB_refine + y

        INF_K = self.linear3(y)
        INF_V = self.linear4(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)

        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        # INF_refine = self.gamma2 * INF_refine + x #送入T分支
        INF_refine = INF_refine + x

        return RGB_refine,INF_refine

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t    
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        #print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    
    keep_index = global_index.gather(dim=1, index=topk_idx)
    
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens
    
    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    #print("finish ce func")

    return tokens_new, keep_index, removed_index


class CEABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search

        self.adap_t = Mixer()
        self.adap2_t = Mixer()

        self.highfuseadapter1 = PSFM(768, 192, 16)  # 通道，降维，双通道
        self.highfuseadapter2 = PSFM(768, 192, 16)

        self.norml1 = norm_layer(dim)
        self.norml2 = norm_layer(dim)

    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None,layer=None):

        B, L, C = x.shape
        xori = x

        if layer in [3,6,9]:

            normrgb = self.norml1(x)
            normt = self.norml1(xi)
            rgbz = normrgb[:, :64, :].permute(0, 2, 1).view(B, -1, 8, 8)
            rgbx = normrgb[:, 64:, :].permute(0, 2, 1).view(B, -1, 16, 16)
            tz = normt[:, :64, :].permute(0, 2, 1).view(B, -1, 8, 8)
            tx = normt[:, 64:, :].permute(0, 2, 1).view(B, -1, 16, 16)

            rgbx, tx = self.highfuseadapter1(rgbx, tx, layer)
            rgbz, tz = self.highfuseadapter1(rgbz, tz, layer)

            rgbx = rgbx.view(B, C, 256).permute(0, 2, 1)
            tx = tx.view(B, C, 256).permute(0, 2, 1)
            rgbz = rgbz.view(B, C, 64).permute(0, 2, 1)
            tz = tz.view(B, C, 64).permute(0, 2, 1)
            fusion0 = torch.cat([rgbz, rgbx], dim=1)
            fusion1 = torch.cat([tz, tx], dim=1)

            # RGB
            x_attn, attn = self.attn(self.norm1(x), mask, True)
            x = x + self.drop_path(x_attn) + self.drop_path(
                self.norm1(fusion0))  #########-------------------------adapter
            # T
            xi_attn, i_attn = self.attn(self.norm1(xi), mask, True)
            xi = xi + self.drop_path(xi_attn) + self.drop_path(
                self.norm1(fusion1))  #########-------------------------adapter
        else:
            x_attn, attn = self.attn(self.norm1(x), mask, True)
            x = x + self.drop_path(x_attn) + self.drop_path(
                self.adap_t(self.norm1(xi)))  #########-------------------------adapter

            xi_attn, i_attn = self.attn(self.norm1(xi), mask, True)
            xi = xi + self.drop_path(xi_attn) + self.drop_path(
                self.adap_t(self.norm1(xori)))  #########-------------------------adapter

        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search,
                                                                                 global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t,
                                                                                    keep_ratio_search,
                                                                                    global_index_searchi,
                                                                                    ce_template_mask)

        xori = x
        if layer in [3,6,9]:
            normrgb1 = self.norml2(x)
            normt1 = self.norml2(xi)
            rgbz1 = normrgb1[:, :64, :].permute(0, 2, 1).view(B, -1, 8, 8)
            rgbx1 = normrgb1[:, 64:, :].permute(0, 2, 1).view(B, -1, 16, 16)
            tz1 = normt1[:, :64, :].permute(0, 2, 1).view(B, -1, 8, 8)
            tx1 = normt1[:, 64:, :].permute(0, 2, 1).view(B, -1, 16, 16)

            rgbx1, tx1 = self.highfuseadapter2(rgbx1, tx1, layer)
            rgbz1, tz1 = self.highfuseadapter2(rgbz1, tz1, layer)

            rgbx1 = rgbx1.view(B, C, 256).permute(0, 2, 1)
            tx1 = tx1.view(B, C, 256).permute(0, 2, 1)
            rgbz1 = rgbz1.view(B, C, 64).permute(0, 2, 1)
            tz1 = tz1.view(B, C, 64).permute(0, 2, 1)
            fusion3 = torch.cat([rgbz1, rgbx1], dim=1)
            fusion4 = torch.cat([tz1, tx1], dim=1)

            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(
                self.norm2(fusion3))  ###-------adapter

            xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(
                self.norm2(fusion4))  ###-------adapter

        else:
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))  ###-------adapter

            xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))  ###-------adapter

        return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
