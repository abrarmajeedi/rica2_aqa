import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck
from .blocks import get_sinusoid_encoding, PhaseDistComposer
from .losses import criterion


class Quality_Score_Head(nn.Module):
    """
    Head for multiple judge quality score
    """

    def __init__(self, input_dim, score_bins, num_random_samples=1, use_stochastic_embd=True, dataset=None):
        super().__init__()

        self.score_bins = score_bins
        self.input_dim = input_dim

        self.common_base = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU()
        )

        self.scoring_head = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, score_bins)
        )

        self.use_stochastic_embd = use_stochastic_embd
        self.phase_composer = PhaseDistComposer(dataset_name=dataset, dim=input_dim//2)

        if self.use_stochastic_embd:
            self.num_random_samples = num_random_samples
            self.mean_var_linear = nn.Sequential(
                                                nn.Linear(input_dim // 2, input_dim // 2),
                                                nn.ReLU(),
                                                nn.Linear(input_dim // 2, input_dim),
                                            )

        self.temperature = nn.Parameter(torch.ones(1) * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(0).unsqueeze(0).expand(*logits.shape)
        return logits / temperature
    
    def forward(self, x, gt_actions= None):
        common_input = self.common_base(x)

        #common_input_shape -> B x phases x dim//2 
        if self.use_stochastic_embd:
            #B x phases x channels*2 
            phase_mean_log_var = self.mean_var_linear(common_input)

            #first C/2 channels are mean, last C/2 channels are log_var
            phase_mean_emb = phase_mean_log_var[:, :, :self.input_dim // 2]
            phase_log_var = phase_mean_log_var[:,:,self.input_dim // 2: ]
            
            phase_var_emb = torch.exp(phase_log_var)

            #shapes = B 
            common_input, global_sqrt_var = self.phase_composer.process_phases(
                                                                    phase_mean_emb = phase_mean_emb,
                                                                    phase_var_emb = phase_var_emb,
                                                                    num_samples = 1 if self.training else self.num_random_samples,
                                                                    gt_actions = gt_actions.detach()
                                                                    ) 
        else:
            # #if phase var emb is none, then it become deterministic
            common_input, _ = self.phase_composer.process_phases(
                                common_input, None, 1, gt_actions.detach()
                                )

        #num_samples x batch_size x 1
        all_sample_output = self.scoring_head(common_input)

        #num_samples x batch_size x 1 --> batch_size x num_samples x 1
        all_sample_output = all_sample_output.permute(1, 0, 2)

        if self.use_stochastic_embd:
            return (
                phase_mean_emb,
                phase_var_emb,
                global_sqrt_var,
                all_sample_output
            )
        else:            
            return all_sample_output



@register_meta_arch("aqa-model")
class AQA_Model(nn.Module):
    """
    Transformer based model for single stage action localization
    """

    def __init__(
        self,
        feat_extractor_type,
        finetune_feat_extractor,
        feat_extractor_weights_path,
        backbone_type,
        input_feat_dim,
        embed_dim,
        conv_dropout,
        neck_type,
        num_layers,
        conv_kernel_size,
        encoder_params,
        decoder_params,
        num_phases,
        score_bins,
        train_cfg,  # other cfg for training
        test_cfg,  # other cfg for testing
        max_seq_len,  # max sequence length in features (used for training)
        frames_per_clip,  # original sequence length in frames
        use_stochastic_embd = False,
        num_random_samples = None
    ):
        super().__init__()

        self.feat_extractor_type = feat_extractor_type
        self.finetune_feat_extractor = finetune_feat_extractor

        self.embed_dim = embed_dim
        self.use_phases =  False

        self.num_phases = num_phases

        self.input_feat_dim = input_feat_dim
        self.max_seq_len = max_seq_len
        self.frames_per_clip = frames_per_clip

        self.score_bins = score_bins

        self.train_cfg = train_cfg


        self.use_stochastic_embd = use_stochastic_embd

        self.num_random_samples = num_random_samples

        if encoder_params["use_abs_pe"]:
            pos_embd = get_sinusoid_encoding(self.max_seq_len, embed_dim) / (
                embed_dim**0.5
            )
            self.register_buffer("pos_embd", pos_embd, persistent=False)
            self.pos_embd = self.pos_embd

        assert feat_extractor_type in ["i3d", None]

        if feat_extractor_type == "i3d":
            self.feat_extractor = make_backbone(
                "i3d",
                **{
                    "I3D_ckpt_path": feat_extractor_weights_path,
                    "finetune": self.finetune_feat_extractor,
                }
            )
        else:
            self.feat_extractor = None

        # backbone network: conv + transformer
        assert backbone_type in ["conv", "convEncoder"]

        if backbone_type == "convEncoder":
            self.backbone = make_backbone(
                "convEncoder",
                **{
                    "n_in": input_feat_dim,  # input feature dimension
                    "n_embd": embed_dim,  # embedding dimension (after convolution)
                    "conv_dropout": conv_dropout,  # dropout rate for conv layers in the initial projection network
                    "n_conv_layers": num_layers["n_conv_layers"],  # number of layers
                    "n_embd_ks": conv_kernel_size,  # conv kernel size of the embedding network
                    "conv_ln": False,  # whether to use layer norm
                    "n_encoder_layers": num_layers[
                        "n_encoder_layers"
                    ],  # number of encoder layers
                    "n_enc_head": encoder_params[
                        "n_encoder_heads"
                    ],  # number of heads in the encoder
                    "attn_pdrop": encoder_params["attn_pdrop"],
                    "proj_pdrop": encoder_params["proj_pdrop"],
                    "path_pdrop": encoder_params["path_pdrop"],
                    "pos_embd": self.pos_embd if encoder_params["use_abs_pe"] else None,
                }
            )
        else:
            self.backbone = None

        if neck_type == "decoder-neck":
            self.neck = make_neck(
                "decoder-neck",
                **{
                    "d_model": embed_dim,
                    "n_heads": decoder_params["n_decoder_heads"],
                    "stride": decoder_params["stride"],
                    "num_decoder_layers": num_layers["n_decoder_layers"],
                    "attn_pdrop": decoder_params["attn_pdrop"],
                    "proj_pdrop": decoder_params["proj_pdrop"],
                    "path_pdrop": decoder_params["path_pdrop"],
                    "xattn_mode": decoder_params["xattn_mode"],
                    "with_ln": decoder_params["with_ln"],
                    "num_phases": num_phases,
                    "query_config": decoder_params["query_config"],
                }
            )
        else:
            self.neck = None


        self.quality_score_head = Quality_Score_Head(
            input_dim=embed_dim,
            score_bins=self.score_bins,
            num_random_samples = self.num_random_samples if self.training else 1,
            use_stochastic_embd = self.use_stochastic_embd,
            dataset=train_cfg["dataset_name"]
        )

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(
        self,
        batched_inputs,
        batched_masks,
        batched_gt_actions,
        video_ids = None,
        curr_epoch = None,
        criterion_args = None
    ):
        if self.feat_extractor is not None:
            batched_inputs = self.feat_extractor(batched_inputs, video_ids)

        # x = B x C x T, mask =  B x 1 x T
        x, masks = self.backbone(batched_inputs, batched_masks, video_ids=video_ids, curr_epoch=curr_epoch)


        if self.neck is not None:
            # output = B x #phases x C
            x, cross_attns = self.neck(x, masks, batched_gt_actions, video_ids=video_ids, curr_epoch=curr_epoch)
        else:
            # B x C x T -> B x T x C
            x = x.permute(0, 2, 1)

        if self.use_stochastic_embd:
            (
                phase_mean_emb,
                phase_var_emb,
                global_sqrt_var_emb,
                all_sample_outputs
            ) = self.quality_score_head(x, batched_gt_actions)
        else:
            #batch_size x num_samples x bins
            all_sample_outputs = self.quality_score_head(x, batched_gt_actions)


        out_dict = {
            "all_sample_outputs": all_sample_outputs,
            "gt_actions": batched_gt_actions,
            "cross_attn": cross_attns if self.neck is not None else None,
        }

        if self.use_stochastic_embd:
            out_dict["phase_mean_emb"] = phase_mean_emb
            out_dict["phase_var_emb"] = phase_var_emb
            out_dict["global_sqrt_var_emb"] = global_sqrt_var_emb


        if criterion_args is not None:
            losses = criterion(out_dict,
                   criterion_args['target'],
                   criterion_args["difficulties"],
                   criterion_args["gt_actions"],
                   loss_weights = criterion_args['loss_weights'],
                   with_dd = criterion_args['with_dd'],
                   three_judge_score_scaling = criterion_args['three_judge_score_scaling'],
                   )
        else:
            losses = None
        
        #remove cross attns from out_dict
        if self.neck is not None:
            out_dict.pop("cross_attn")
        return out_dict, losses