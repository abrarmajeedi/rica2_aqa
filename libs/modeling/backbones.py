import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (TransformerBlock, MaskedConv1D,
                     LayerNorm)
from .i3d import I3D


@register_backbone("convEncoder")
class ConvEncoderBackbone(nn.Module):
    """
        A backbone that with convs and encoder blocks
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        n_conv_layers,
        conv_dropout,
        conv_ln,            # if to use layernorm
        n_encoder_layers,
        n_enc_head,
        attn_pdrop,
        proj_pdrop,
        path_pdrop,
        pos_embd
    ):
        super().__init__()
        self.n_in = n_in

        self.conv_dropout = conv_dropout
        self.n_encoder_layers = n_encoder_layers
        self.relu = nn.ReLU(inplace=True)

        self.pos_embd =  pos_embd

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(n_conv_layers):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not conv_ln)
                )
            )
            if conv_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())
        
        self.encoder_blocks = nn.ModuleList([TransformerBlock(n_embd= n_embd,
                                                    n_head = n_enc_head,
                                                    n_hidden= n_embd,
                                                    n_out = n_embd,
                                                    attn_pdrop=attn_pdrop,
                                                    proj_pdrop=proj_pdrop,
                                                    path_pdrop=path_pdrop,
                                                    use_rel_pe = False, #only for local attn, not applicable here
                                                    ) for _ in range(n_encoder_layers)])
        
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask, video_ids = None, curr_epoch = None):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

            if idx != len(self.embd) - 1:
                x = nn.Dropout(self.conv_dropout)(x)                                               
            
        B, C, T = x.size()

        if self.pos_embd is not None:
            x = x + self.pos_embd[:, :, :T].cuda() * mask.to(x.dtype)
        
        
        for idx in range(self.n_encoder_layers):
            x, mask = self.encoder_blocks[idx](x, mask)
        
        return x, mask



@register_backbone("i3d")
class I3D_feat_extractor(nn.Module):
    def __init__(self, I3D_ckpt_path, finetune):
        super(I3D_feat_extractor, self).__init__()
        self.i3d = I3D(num_classes=400, modality='rgb', dropout_prob=0.5)

        if I3D_ckpt_path is not None:
            print("loading I3D weights from: ", I3D_ckpt_path)
            self.i3d.load_state_dict(torch.load(I3D_ckpt_path))
        
        self.finetune =  finetune
        self.se = False
        
        self.avg_pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        

    def get_feature_dim(self):
        return self.i3d.get_logits_dim()

    def forward(self, videos, video_ids = None):

        if videos.dim() == 6:
            #B x Nwin x T x C x H x W
            B, N_Win, T, C, H, W = videos.size()

            #B x N_Win x T x C x H x W ->  B*N_Win x T x C x H x W
            videos_reshaped = videos.reshape(B*N_Win, T, C, H, W)
            
            #B*N_Win, T, C, H, W -> B*N_Win, C, T, H, W
            videos_tensor = videos_reshaped.permute(0, 2, 1, 3, 4)

        if videos.dim() == 7:
            #B x Nwin x T x C x H x W
            B, N_Win, crops, T, C, H, W = videos.size()

            #B x N_Win x T x C x H x W ->  B*N_Win*crops x T x C x H x W
            videos_reshaped = videos.reshape(B*N_Win*crops, T, C, H, W)

            #B*N_Win*crops, T, C, H, W -> B*N_Win*crops, C, T, H, W
            videos_tensor = videos_reshaped.permute(0, 2, 1, 3, 4)

        if not self.finetune:
            with torch.no_grad():
                self.i3d.eval()
                video_feature = self.i3d(videos_tensor)
        else:
            video_feature = self.i3d(videos_tensor)


        #Video -> B*N_Win, C
        video_feature = self.avg_pool(video_feature).squeeze()


        if videos.dim() == 6:
            #Split into batches (B x T x C), because N_Win is the final T
            batch_feats =  video_feature.reshape(B, N_Win, -1)
        
        if videos.dim() == 7:
            batch_feats = video_feature.reshape(B, N_Win, crops, -1)

            batch_feats = batch_feats.mean(axis=2)

        #B x T x C -> B x C x T
        batch_feats = batch_feats.permute(0, 2, 1)

        return batch_feats


