import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat


"""
ranking and sparsity loss, modified from TPT
Uses broadcasted operations to calculate the the loss in a fully vectorized manner.
https://github.com/baiyang4/aqa_tpt/blob/cf6d1631ec53c676f108926fc5480afe7c47ff56/train_pairencode1_decoder_1selfatt_self8head_ffn_sp_new.py#L289
"""
def get_att_loss(logits_all, gt_actions,ranking_loss_wt, sparsity_loss_wt, hinge_loss):

    # logits_all: B, Q, T
    B, Q, T = logits_all.size()
    #gt_actions One hot encoded: B, Q

    logits_all = logits_all.permute(0, 2, 1) #.transpose(-1,-2)
    softmax_dim = logits_all.shape[1]
    temp_idx = repeat(torch.arange(1, softmax_dim + 1), 't -> b t k', b=logits_all.shape[0], k=logits_all.shape[-1]).float().to(logits_all.device)
    cluster_mean = (logits_all * temp_idx).sum(1)
    var = (torch.abs(temp_idx - repeat(cluster_mean, 'b k -> b t k', t=softmax_dim)) * logits_all).sum(1) 
    
    if ranking_loss_wt == 0.0:
        return 0.0, var.mean()
    
    # Extract active clusters based on gt_actions for all samples in batch
    active_indices = gt_actions.nonzero(as_tuple=True)  # Indices where gt_actions is 1
    active_clusters = cluster_mean[active_indices]  # Extract active clusters

    # Calculate hinge loss for ranking the active clusters in a fully vectorized manner
    cluster_counts = gt_actions.sum(dim=1).long()  # Number of active clusters per batch
    max_clusters = cluster_counts.max().item()

    # Pad active clusters to have the same length per batch
    padded_active_clusters = torch.zeros((B, max_clusters), device=cluster_mean.device)
    mask = torch.arange(max_clusters, device=cluster_mean.device).unsqueeze(0) < cluster_counts.unsqueeze(1)
    padded_active_clusters[mask] = active_clusters

    # Create shifted versions of the active clusters for comparison
    current_clusters = padded_active_clusters[:, :-1]  # B, max_clusters - 1
    next_clusters = padded_active_clusters[:, 1:]  # B, max_clusters - 1
    valid_pairs_mask = mask[:, :-1] & mask[:, 1:]  # Mask to identify valid pairs for loss calculation

    # Apply the valid pairs mask to current and next clusters
    valid_current_clusters = current_clusters[valid_pairs_mask]
    valid_next_clusters = next_clusters[valid_pairs_mask]

    # Calculate hinge loss for valid pairs
    ones = torch.ones_like(valid_current_clusters)
    loss = hinge_loss(valid_next_clusters, valid_current_clusters, ones).mean()

    # Add boundary conditions (first and last clusters only once per batch)
    first_clusters = padded_active_clusters[:, 0]  # B
    last_clusters = padded_active_clusters[torch.arange(B), cluster_counts - 1]  # B
    boundary_mask = cluster_counts > 0  # Mask to identify batches with at least one active cluster

    loss += hinge_loss(first_clusters[boundary_mask], torch.ones_like(first_clusters[boundary_mask]), torch.ones_like(first_clusters[boundary_mask])).sum()
    loss += hinge_loss(torch.ones_like(last_clusters[boundary_mask]) * softmax_dim, last_clusters[boundary_mask], torch.ones_like(last_clusters[boundary_mask])).sum()

    return loss, var.mean()



def criterion(model_output, target, difficulties, inp_gt_actions, loss_weights, with_dd=True, three_judge_score_scaling=False):

    batched_gt_judge_scores = target

    #B, samples, 1
    batched_pred_quality_scores = model_output["all_sample_outputs"]

    difficulties = difficulties.cuda()

    # comment these out when training jigsaws?? TODO
    if with_dd:
        batched_gt_judge_scores = batched_gt_judge_scores * difficulties
        batched_pred_quality_scores = batched_pred_quality_scores * difficulties.unsqueeze(-1).unsqueeze(-1)

    if three_judge_score_scaling:
        batched_gt_judge_scores = batched_gt_judge_scores * 3.0
        batched_pred_quality_scores = batched_pred_quality_scores * 3.0

    if loss_weights["phase_vib"] != 0:

        gt_actions = inp_gt_actions.detach()
        gt_actions = gt_actions.reshape(-1)

        phase_mean_emb, phase_var_emb = model_output["phase_mean_emb"], model_output["phase_var_emb"]
        
        batch_size, num_phases, channels = phase_mean_emb.shape
        phase_mean_emb = phase_mean_emb.reshape(-1, channels)
        phase_var_emb = phase_var_emb.reshape(-1, channels)

        #mask out the non-action phases
        if num_phases > 1:
            phase_mean_emb = phase_mean_emb[gt_actions.nonzero().squeeze().long(),:]
            phase_var_emb = phase_var_emb[gt_actions.nonzero().squeeze().long(),:]

        #B*phases x channels -> B*phases
        phase_vib_loss = torch.sum(torch.pow(phase_mean_emb, 2) + phase_var_emb - torch.log(phase_var_emb) - 1.0, dim=1) * 0.5

        phase_vib_loss = phase_vib_loss.sum()
        phase_vib_loss = phase_vib_loss / (batch_size * channels)
    

    # batched_pred_quality_score_logits shape -> batch_size x num_samples x 1
    # batched_gt_judge_score_class shape -> B x 1
    B, num_samples, bins = batched_pred_quality_scores.shape

    if loss_weights["loss"] == "mse":
        ground_truth_expanded =  batched_gt_judge_scores.unsqueeze(1).expand(-1, num_samples)

        batched_pred_quality_scores_flat = batched_pred_quality_scores.reshape(-1)
        ground_truth_flat = ground_truth_expanded.reshape(-1)

        quality_score_loss = nn.MSELoss(reduction='mean')(batched_pred_quality_scores_flat, ground_truth_flat)

    elif loss_weights["loss"] =="xentropy":
        ground_truth_expanded =  batched_gt_judge_scores.unsqueeze(1).expand(-1, num_samples)
        
        batched_pred_quality_scores_flat = batched_pred_quality_scores.reshape(B, bins)
        ground_truth_flat = ground_truth_expanded.reshape(B).long()
        quality_score_loss = nn.CrossEntropyLoss(reduction="mean")(batched_pred_quality_scores_flat, ground_truth_flat)
    else:
        raise ValueError("Invalid loss function")

    if loss_weights["ranking"] > 0.0 or loss_weights["sparsity"] > 0.0:
        # bz x heads x num_queries x T
        # Flatten the logits and gt_actions across batch, heads, and decoder layers dimensions
        batch_size, num_heads, num_queries, num_clips = model_output["cross_attn"][0].shape
        num_decoder_layers = len(model_output["cross_attn"])

        # Concatenate across decoder layers and heads to flatten
        logits_all_flat = torch.cat([model_output["cross_attn"][decoder_layer_idx][:, head_idx, :, :]
                                    for decoder_layer_idx in range(num_decoder_layers)
                                    for head_idx in range(num_heads)], dim=0)  # Shape: (B * num_heads * num_decoder_layers), Q, T

        # Repeat the ground truth actions for each head and decoder layer
        gt_actions_flat = inp_gt_actions.detach().repeat(num_heads * num_decoder_layers, 1)  # Shape: (B * num_heads * num_decoder_layers), Q

        ranking_loss, sparsity_loss = get_att_loss(logits_all_flat, 
                                           gt_actions_flat, 
                                           ranking_loss_wt=loss_weights["ranking"],
                                           sparsity_loss_wt=loss_weights["sparsity"],
                                           hinge_loss=nn.MarginRankingLoss(1.0))
    
    
    loss_dict, final_loss = {}, 0.0
    valid_loss_keys = ["quality_score", "phase_vib", "ranking", "sparsity"]
 
    for key, value in loss_weights.items():
        if key in valid_loss_keys and isinstance(value, (int, float)) and value != 0:
            loss_dict[f"{key}_loss"] = value * locals()[f"{key}_loss"]
            final_loss += loss_dict[f"{key}_loss"]

    loss_dict["final_loss"] = final_loss

    return loss_dict
