import os
import shutil
import time
import json
import pickle
from typing import Dict

import numpy as np

import torch
import torch.nn as nn

@torch.no_grad()
def update_metric_dict_with_model_output(metric_dict, model_output, gt_scores, difficulties, is_val, cfg=None):
    # B, samples, 1
    all_sample_outputs = model_output["all_sample_outputs"].detach()

    with_dd = cfg["dataset"]["with_dd"]
    
    batched_pred_quality_scores = get_pred_scores(all_sample_outputs, cfg).detach().cpu().numpy()
    batched_gt_scores = gt_scores.detach().cpu().numpy()

    if with_dd:
        batched_pred_quality_scores = batched_pred_quality_scores * difficulties.detach().cpu().numpy()

    metric_dict.update("pred_scores", batched_pred_quality_scores)
    metric_dict.update("gt_scores", batched_gt_scores)

    if is_val:
        if "global_sqrt_var_emb" in model_output.keys():
            metric_dict.update("global_sqrt_var_emb", model_output["global_sqrt_var_emb"].detach().cpu().numpy())
            return
                

def get_pred_scores(all_sample_outputs, cfg):
    #B, samples, 1:  all_sample_outputs.size()

    if cfg["dataset_name"] == "jigsaws":
        batched_pred_quality_scores = all_sample_outputs.mean(1).squeeze()
        if cfg["dataset"]["six_item_score_scaling"]:
            batched_pred_quality_scores = batched_pred_quality_scores * 6.0
    else:
        batched_pred_quality_scores = all_sample_outputs.mean(1).squeeze()
        if cfg["dataset"]["three_judge_score_scaling"]:
            batched_pred_quality_scores = batched_pred_quality_scores * 3.0

    return batched_pred_quality_scores 
