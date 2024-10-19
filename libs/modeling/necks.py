import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .models import register_neck
from .blocks import TransformerDecoder, LayerNorm

@register_neck("decoder-neck")
class Decoder(nn.Module):
    def __init__(self, d_model=512,
                 n_heads=8,
                 stride=1,
                 num_decoder_layers=6,
                 attn_pdrop=0.0,
                 proj_pdrop=0.0,
                 path_pdrop=0.0,
                 num_phases=-1,
                 xattn_mode='affine',
                 with_ln=True,
                 use_rel_pe=False,
                 query_config=None):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.layers.append(
                TransformerDecoder(
                    embd_dim = d_model,
                    kv_dim = d_model,
                    stride=stride,
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                    use_rel_pe=use_rel_pe
                )
            )

        self.ln_out = LayerNorm(d_model) if with_ln else None

        assert num_phases > 0, "Number of phases must be > 0"

        self.num_phases = num_phases

        self._reset_parameters()
        text_queries_emb_dim = query_config['text_queries_emb_dim']

        self.d_model = d_model
        self.n_heads = n_heads
        self.query_config = query_config

        text_embd = torch.from_numpy(np.load(query_config['text_embeddings_path']))
        self.text_queries = nn.Embedding.from_pretrained(text_embd.squeeze(), 
                                                    freeze=query_config['freeze_text_embeddings'])
        self.text_queries_projection = nn.Linear(
            in_features=text_queries_emb_dim, out_features=d_model, bias=True) 
    
    def _get_queries(self):
        # num phases x C
        text_queries = self.text_queries.weight

        # num phases x C -> num phases x d_model
        queries = self.text_queries_projection(text_queries)
        
        return queries

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None, video_ids=None, curr_epoch=None):
        layerwise_cross_attn = [] 
        for layer in self.layers:
            q, q_mask, cross_attn = layer(q, q_mask, kv, kv_mask, kv_size,
                                          video_ids=video_ids, curr_epoch=curr_epoch)
            layerwise_cross_attn.append(cross_attn)
            
        q = self.ln_out(q)

        return q, q_mask, layerwise_cross_attn

    
    def forward(self, vid_embd, vid_masks, batched_gt_actions,  video_ids=None, curr_epoch=None):

        """
        vid_embd: B, C, T
        vid_masks: B, T
        batched_gt_actions: B, phases
        """
        B, C, T = vid_embd.shape
        
        #phases x d_model
        queries = self._get_queries()
        
        #phases x d_model -> B x phases x d_model
        queries = queries.repeat(B, 1, 1)

        #B x phases x d_model -> B x d_model x phases
        queries = queries.permute(0, 2, 1)

        #B x 1 x Phases
        query_masks = batched_gt_actions.unsqueeze(1).bool().detach()

        out, out_masks, cross_attns =  self._forward(queries, query_masks, vid_embd, vid_masks, kv_size=T,
                                        video_ids=video_ids, curr_epoch=curr_epoch)

        #B x d_model x phases -> B x phases x d_model
        out = out.permute(0, 2, 1)
        
        return out, cross_attns
