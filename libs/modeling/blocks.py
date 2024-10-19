import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_

def reparameterize(B_mu, B_sigma, num_samples=1):
    expanded_mu = B_mu.expand(num_samples, *B_mu.shape)
    expanded_sigma = B_sigma.expand(num_samples, *B_sigma.shape)
    norm_v = torch.randn_like(expanded_mu).detach()
    # reparameterization trick
    samples = expanded_mu + expanded_sigma * norm_v
    return samples

class PhaseDistComposer(nn.Module):
    def __init__(self, dataset_name='finediving', dim = 256):
        super().__init__()
        self.dataset_name = dataset_name
        if dataset_name == 'finediving':
            self.level_0_to_1_index = [torch.tensor(x).detach().long() for x in [[0,1,2,3,4,5,6],[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],[28]]]
        elif dataset_name =="mtl_aqa":
            self.level_0_to_1_index = [torch.tensor(x).detach().long() for x in [[0,1,2,3,4],[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],[23]]]
        elif dataset_name == "needle_passing":
            self.level_0_to_1_index = [torch.tensor(x).detach().long() for x in [[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]]
        elif dataset_name == "knot_tying":
            self.level_0_to_1_index = [torch.tensor(x).detach().long() for x in [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]]
        elif dataset_name == "suturing":
            self.level_0_to_1_index = [torch.tensor(x).detach().long() for x in [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]]
        
        self.level_2_mlp = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )

        self.level_3_mlp = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )
        
    def get_one_sample(self, phase_mean_emb, phase_var_emb, gt_actions):
        #B, Phases, C
        if phase_var_emb is None:
            level_0_samples = phase_mean_emb
        else:
            phase_sqrt_var_emb = torch.sqrt(phase_var_emb)
            level_0_samples = reparameterize(phase_mean_emb, phase_sqrt_var_emb, num_samples=1).squeeze(0)

        B, num_phases, C = phase_mean_emb.shape

        level_1_inputs_separated = [level_0_samples[:,node_input_idx,:] for node_input_idx in self.level_0_to_1_index]

        level_1_presence_separated = [gt_actions[:,node_input_idx].unsqueeze(-1) for node_input_idx in self.level_0_to_1_index]

        #num nodes, B x phases going into the node x C
        level_1_inputs_filtered_by_presence = [level_1_inputs_separated[node_num] * level_1_presence_separated[node_num] for node_num in range(len(level_1_inputs_separated))]

        #num nodes, B x C
        level_1_inputs_summed_separate = [torch.sum(level_1_inputs_filtered_by_presence[node_num], dim=1) 
                                          / torch.sum(level_1_presence_separated[node_num], dim=1) 
                                          for node_num in range(len(level_1_inputs_filtered_by_presence))]

        #num nodes, B x C
        level_2_outputs = [self.level_2_mlp(x) for x in level_1_inputs_summed_separate]

        #num_nodes x B x C
        level_2_outputs = torch.stack(level_2_outputs, dim=0) 

        #level_3_input is the same as output since we have more decoding heads later
        level_3_output = torch.sum(level_2_outputs, dim=0) / level_2_outputs.shape[0]

        #check if nan in level_3_output
        if torch.isnan(level_3_output).any():
            import ipdb; ipdb.set_trace()

        return level_3_output

    
    def process_phases(self, phase_mean_emb, phase_var_emb, num_samples, gt_actions):
        all_samples = []
        for i in range(num_samples):
            one_sample_per_item = self.get_one_sample(phase_mean_emb, phase_var_emb, gt_actions)
            all_samples.append(one_sample_per_item)

        #num_samples, B, C
        all_samples = torch.stack(all_samples, dim=0)

        if phase_var_emb is None:
            return all_samples, None
        
        masked_phase_var_emb = phase_var_emb * gt_actions.unsqueeze(-1)

        #global variance
        global_masked_var = torch.sum(masked_phase_var_emb, dim=1) / torch.sum(gt_actions, dim=1).unsqueeze(-1).detach()

        #global sigma
        global_sigma = torch.sqrt(global_masked_var)

        return all_samples, global_sigma


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

class MaskedAvgPool1D(nn.Module):
    """
    Masked 1D average pooling
    """
    def __init__(self):
        super(MaskedAvgPool1D, self).__init__()

    def forward(self, x, mask):
        x_sum = torch.sum(x * mask.float(), dim=-1, keepdim=True)
        n = torch.sum(mask, dim=-1, keepdim=True)
        x = x_sum / n
        
        return x


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)

class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask
    NOTE: This implementation supports
        - global and local self-attention
        - (global) cross-attention

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        q_dim=None,         # query dimension
        kv_dim=None,        # key / value dimension
        out_dim=None,       # output dimension
        n_heads=4,          # number of attention heads
        window_size=0,      # local attention window size (0 for global attention)
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
        use_rel_pe=False,   # whether to apply relative position encoding
    ):
        super(MaskedMHA, self).__init__()

        assert embd_dim % n_heads == 0
        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / math.sqrt(math.sqrt(self.n_channels))
        self.out_dim = out_dim

        self.query = nn.Conv1d(q_dim, embd_dim, 1)
        self.key = nn.Conv1d(kv_dim, embd_dim, 1)
        self.value = nn.Conv1d(kv_dim, embd_dim, 1)
        self.proj = nn.Conv1d(embd_dim, out_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # local attention window size
        assert window_size == 0 or window_size % 2 == 1
        self.window_size = window_size
        self.stride = window_size // 2

        # masks for local attention (left / right paddings)
        l_mask = torch.ones(self.stride, self.stride + 1).tril().flip(dims=(0,))
        r_mask = torch.ones(self.stride, self.stride + 1).tril().flip(dims=(1,))
        self.register_buffer('l_mask', l_mask.bool(), persistent=False)
        self.register_buffer('r_mask', r_mask.bool(), persistent=False)

        # relative position encoding for local attention
        if window_size > 0 and use_rel_pe:
            self.rel_pe = nn.Parameter(torch.zeros(n_heads, 1, window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / embd_dim) ** 0.5)
        else:
            self.rel_pe = None

    def _chunk(self, x, size):
        """
        Convert feature sequence into temporally overlapping chunks.

        Args:
            x (float tensor, (n, t, d)): feature sequence.
            size (int): chunk size.

        Returns:
            x (float tensor, (n, k, s, d)): chunked features.
        """
        n, t, d = x.size()
        assert (t + self.stride - size) % self.stride == 0
        n_chunks = (t + self.stride - size) // self.stride

        chunk_size = (n, n_chunks, size, d)
        chunk_stride = (x.stride(0), self.stride * x.stride(1), *x.stride()[1:])
        x = x.as_strided(size=chunk_size, stride=chunk_stride)

        return x

    def _query_key_matmul(self, q, k):
        """
        Chunk-wise query-key product.

        Args:
            q (float tensor, (n, t, d)): query tensor.
            k (float tensor, (n, t, d)): key tensor.

        Returns:
            attn (float tensor, (n, t, w)): unnormalized attention scores.
        """
        assert q.size() == k.size()
        n, t, _ = q.size()
        w, s = self.window_size, self.stride

        # chunk query and key tensors: (n, t, d) -> (n, t // s - 1, 2s, d)
        q_chunks = self._chunk(q.contiguous(), size=2 * s)
        k_chunks = self._chunk(k.contiguous(), size=2 * s)
        n_chunks = q_chunks.size(1)

        # chunk-wise attention scores: (n, t // s - 1, 2s, 2s)
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (q_chunks, k_chunks))

        # shift diagonals into columns: (n, t // s - 1, 2s, w)
        chunk_attn = F.pad(chunk_attn, (0, 0, 0, 1))
        chunk_attn = chunk_attn.view(n, n_chunks, 2 * s, w)

        # fill in the overall attention matrix: (n, t // s, s, w)
        attn = chunk_attn.new_zeros(n, t // s, s, w)
        attn[:, :-1, :, s:] = chunk_attn[:, :, :s, :s + 1]
        attn[:, -1, :, s:] = chunk_attn[:, -1, s:, :s + 1]
        attn[:, 1:, :, :s] = chunk_attn[:, :, -(s + 1):-1, s + 1:]
        attn[:, 0, 1:s, 1:s] = chunk_attn[:, 0, :s - 1, w - (s - 1):]
        attn = attn.view(n, t, w)

        # mask invalid attention scores
        attn[:, :s, :s + 1].masked_fill_(self.l_mask, float('-inf'))
        attn[:, -s:, -(s + 1):].masked_fill_(self.r_mask, float('-inf'))

        return attn

    def _attn_normalize(self, attn, mask):
        """
        Normalize attention scores over valid positions.

        Args:
            attn (float tensor, (bs, h, t, w)): unnormalized attention scores.
            mask (bool tensor, (bs, t, 1)): mask (1 for valid positions).

        Returns:
            attn (float tensor, (bs, h, t, w)): normalized attention map.
        """
        bs, h, t, w = attn.size()

        # inverse mask (0 for valid positions, -inf for invalid ones)
        inv_mask = torch.logical_not(mask)
        inv_mask_float = inv_mask.float().masked_fill(inv_mask, -1e4)

        # additive attention mask: (bs, t, w)
        attn_mask = self._query_key_matmul(
            torch.ones_like(inv_mask_float), inv_mask_float
        )
        attn += attn_mask.view(bs, 1, t, w)

        # normalize
        attn = F.softmax(attn, dim=-1)

        # if all key / value positions in a local window are invalid
        # (i.e., when the query position is invalid), softmax returns NaN.
        # Replace NaNs with 0
        attn = attn.masked_fill(inv_mask.unsqueeze(1), 0.0)

        return attn

    def _attn_value_matmul(self, attn, v):
        """
        Chunk-wise attention-value product.

        Args:
            attn (float tensor, (n, t, w)): attention map.
            v (float tensor, (n, t, d)): value tensor.

        Returns:
            out (float tensor, (n, t, d)): attention-weighted sum of values.
        """
        n, t, d = v.size()
        w, s = self.window_size, self.stride

        # chunk attention map: (n, t, w) -> (n, t // s, s, w)
        attn_chunks = attn.view(n, t // s, s, w)

        # shift columns into diagonals: (n, t // s, s, 3s)
        attn_chunks = F.pad(attn_chunks, (0, s))
        attn_chunks = attn_chunks.view(n, t // s, -1)[..., :-s]
        attn_chunks = attn_chunks.view(n, t // s, s, 3 * s)

        # chunk value tensor: (n, t + 2s, d) -> (n, t // s, 3s, d)
        v = F.pad(v, (0, 0, s, s))
        v_chunks = self._chunk(v.contiguous(), size=3 * s)

        # chunk-wise attention-weighted sum: (n, t // s, s, d)
        out = torch.einsum('bcwd,bcdh->bcwh', (attn_chunks, v_chunks))
        out = out.view(n, t, d)

        return out

    def forward(self, q, k=None, v=None, kv_mask=None, kv_size=None,
                video_ids=None, layer_idx=None, curr_epoch=None, q_mask=None):
        """
        Args:
            q (float tensor, (bs, c, t1)): query feature sequence.
            k (float tensor, (bs, c, t2)): key feature sequence.
            v (float tensor, (bs, c, t2)): value feature sequence.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask.
            kv_size (int tensor, (bs,)): number of times to repeat each sample.
        """
        bs, c = q.size(0), self.embd_dim
        h, d, w = self.n_heads, self.n_channels, self.window_size

        if k is None:
            k = q
        if v is None:
            v = k

        # if mask is not given, assume all positions are valid
        if kv_mask is None:
            kv_mask = torch.ones_like(k[:, :1], dtype=torch.bool)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # repeat query to match the size of key / value
        if kv_size is not None and k.size(0) != bs:
            q = q.repeat_interleave(kv_size, dim=0)
            bs = q.size(0)

        if self.window_size > 0:
            q = q.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            k = k.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            v = v.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)

            # attention scores: (bs * h, t, w)
            attn = self._query_key_matmul(q * self.scale, k * self.scale)
            attn = attn.view(bs, h, -1, w)
            if self.rel_pe is not None:
                attn += self.rel_pe

            # normalized attention map: (bs, h, t, w)
            attn = self._attn_normalize(attn, kv_mask.transpose(1, 2))
            attn = self.attn_drop(attn)
            attn = attn.view(bs * h, -1, w)

            # attention-weighted sum of values: # (bs * h, t, d)
            q = self._attn_value_matmul(attn, v)
            q = q.view(bs, h, -1, d)
        else:
            q = q.view(bs, h, d, -1).transpose(2, 3)
            k = k.view(bs, h, d, -1)
            v = v.view(bs, h, d, -1).transpose(2, 3)

            attn = (q * self.scale) @ (k * self.scale)      # (bs, h, t1, t2)
            attn = attn.masked_fill(
                mask=torch.logical_not(kv_mask[:, :, None, :]),
                value=float('-inf'),
            )
            attn = F.softmax(attn, dim=-1)
            
            ret_attn = attn.clone()
            
            attn = self.attn_drop(attn)
            q = attn @ v                                    # (bs, h, t1, d)

        q = q.transpose(2, 3).reshape(bs, c, -1)            # (bs, c, t1)
        out = self.proj_drop(self.proj(q))

        return out, ret_attn * q_mask.unsqueeze(-1).expand(attn.shape).requires_grad_(False)


class AttNPool1D(nn.Module):
    def __init__(self, embd_dim, n_heads=4):
        super(AttNPool1D, self).__init__()

        self.pool = MaskedAvgPool1D()
        self.attn = MaskedMHA(embd_dim, n_heads=n_heads)

    def forward(self, x, mask):
        x_mean = self.pool(x, mask)
        h = torch.cat((x_mean, x), dim=-1)
        mask = torch.cat((mask[..., :1], mask), dim=-1)

        pool = self.attn(h, kv_mask=mask)[..., :1]
        x = torch.cat((pool, x), dim=-1)
        
        return x, mask



class ConvXAttNLayer(nn.Module):
    """
    Multi Head Conv Cross Attention with mask

    With current implementation, the downpsampled features will be aligned with
    every s+1 time steps, where s is the down-sampling stride. This allows us
    to easily interpolate the corresponding position encoding.
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        kv_dim,             # key / value dimension
        out_dim=None,       # output dimension
        stride=1,           # convolution stride
        n_heads=4,          # number of attention heads
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
        use_offset=False,   # whether to add offsets to down-sampled points
    ):
        super(ConvXAttNLayer, self).__init__()

        self.use_conv = stride > 0
        if self.use_conv:
            assert stride == 1 or stride % 2 == 0
            kernel_size = stride + 1 + use_offset if stride > 1 else 3
            padding = (kernel_size - 1) // 2

            self.q_conv = MaskedConv1D(
                embd_dim, embd_dim,
                kernel_size=kernel_size,
                stride=stride, padding=padding,
                groups=embd_dim, bias=False,
            )
            self.q_norm = LayerNorm(embd_dim)
        else:
            self.q_conv = self.q_norm = None

        if out_dim is None:
            out_dim = embd_dim

        # cross-attention
        self.xattn = MaskedMHA(
            embd_dim, 
            kv_dim=kv_dim, out_dim=out_dim,
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop, proj_pdrop=proj_pdrop
        )

    def forward(self, q, q_mask, kv, kv_mask, kv_size=None,
                video_ids=None, layer_idx=None, curr_epoch=None):
        if self.use_conv:
            q, q_mask = self.q_conv(q, q_mask)
            q = self.q_norm(q)

        out, cross_attn = self.xattn(q, kv, None, kv_mask, kv_size,
                                     video_ids=video_ids, layer_idx=layer_idx, 
                                     curr_epoch=curr_epoch, q_mask = q_mask)
        if kv_size is not None and out.size(0) != q_mask.size(0):
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return out, q_mask, cross_attn


class FFN(nn.Module):
    """
    Feed Forward Network (MLP) in Transformer.
    """
    def __init__(self, channels, expansion=4, pdrop=0.0):
        super(FFN, self).__init__()

        self.fc = nn.Conv1d(channels, channels * expansion, 1)
        self.actv = nn.GELU()
        self.proj = nn.Conv1d(channels * expansion, channels, 1)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.dropout(self.actv(self.fc(x)))
        x = self.dropout(self.proj(x))

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder (w/o self-attention).
    (optional depth-wise conv -> xattn -> FFN)
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        kv_dim,             # key / value dimension
        stride=1,           # convolution stride (0 if disable convs)
        n_heads=4,          # number of attention heads
        window_size=0,      # MHA window size (0 for global attention)
        expansion=4,        # expansion factor for FFN
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (affine | adaln)
        use_offset=False,   # whether to add offsets to down-sampled points
        use_rel_pe=False,   # whether to apply relative position encoding
    ):
        super(TransformerDecoder, self).__init__()

        # cross-attention
        assert xattn_mode in ('affine', 'adaln')
        self.xattn = ConvXAttNLayer(
            embd_dim, kv_dim,
            out_dim=embd_dim * 2,
            stride=stride, n_heads=n_heads,
            attn_pdrop=attn_pdrop, proj_pdrop=path_pdrop,
            use_offset=use_offset
        )
        self.ln_xattn_q = LayerNorm(embd_dim)
        self.ln_xattn_kv = LayerNorm(kv_dim)

        if xattn_mode == 'adaln':
            self.adaln = LayerNorm(embd_dim, affine=False)
        else:
            self.adaln = None

        # FFN
        self.ffn = FFN(embd_dim, expansion, proj_pdrop)
        self.ln_ffn = LayerNorm(embd_dim)

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_ffn = AffineDropPath(embd_dim, drop_prob=path_pdrop)
        else:
            self.drop_path_ffn = nn.Identity()

    def forward(self, q, q_mask, kv, kv_mask, kv_size=None,
                video_ids=None, curr_epoch=None):
        if q_mask is None:
            q_mask = torch.ones_like(q[:, :1], dtype=torch.bool)

        q = q * q_mask.float()


        # cross-attention (optionally with depth-wise conv)
        out, q_mask, cross_attn = self.xattn(
            self.ln_xattn_q(q), q_mask, self.ln_xattn_kv(kv), kv_mask, kv_size,
            video_ids=video_ids, curr_epoch=curr_epoch
        )
        if kv_size is not None and q.size(0) != out.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
               
        q = q * q_mask.float()
        weight, bias = out.split(q.size(1), dim=1)

        if self.adaln is not None:
            q = self.adaln(q)

        q = q * weight + bias

        # FFN
        out = self.ffn(self.ln_ffn(q)) * q_mask.float()
        q = q + self.drop_path_ffn(out)

        return q, q_mask, cross_attn


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init=1.0):
        """
        init_value : initial value for the scalar
        """
        super(Scale, self).__init__()

        self.scale = nn.Parameter(torch.as_tensor(init, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    x = x.div(keep_prob) * mask

    return x


class DropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks)
    with a per channel scaling factor (and zero init).

    https://arxiv.org/pdf/2103.17239.pdf
    """
    def __init__(self, dim, drop_prob=0.0, init_scale=1e-4):
        super(AffineDropPath, self).__init__()

        self.scale = nn.Parameter(init_scale * torch.ones((1, dim, 1)))
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kernel_size=3,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        expansion_factor=2,    # expansion factor of feat dims
        n_out=None,            # output dimension, if None, set to input dim
        act_layer=nn.ReLU,     # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

         # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask



class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False       # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # only valid for local attention
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask
