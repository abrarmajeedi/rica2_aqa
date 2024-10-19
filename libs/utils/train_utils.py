import os
import time
import sys
import numpy as np
import random
from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR, LinearWarmupNoDecayLR

from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from .metrics import evaluate, MetricsDict
from .preprocessing import preprocessing
from .postprocessing import update_metric_dict_with_model_output
from collections import defaultdict


from torch.profiler import profile, record_function, ProfilerActivity

class Logger(object):
    def __init__(self, log_file):
        self.terminal =  sys.stdout
        self.log_file = log_file

    def write(self, message):
        #import ipdb; ipdb.set_trace()
        message = str(message)
        #pprint(message, stream = sys.__stdout__)
        self.terminal.write(message)
        #print(message + "\n", file=sys.__stdout__, end = "")
        with open(self.log_file, 'a') as f:
            f.write(message)
    def flush(self):
        self.terminal.flush()

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_top_3_blocks(m):
    classname = m.__class__.__name__
    if (
        classname.find('conv3d_1a_7x7') != -1 or 
        classname.find('conv3d_2b_1x1') != -1 or 
        classname.find('conv3d_2c_3x3') != -1 or 
        classname.find('mixed_3b') != -1 or 
        classname.find('mixed_3c') != -1 or 
        classname.find('mixed_4b') != -1
        ):
        m.requires_grad = False
        m.eval()


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # print("Determininstic is off")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134

    low_lr_decay = set()
    low_lr_no_decay = set()
    
    med_lr_decay = set()
    med_lr_no_decay = set()

    high_lr_decay = set()
    high_lr_no_decay = set()

    whitelist_weight_modules = (torch.nn.Linear, nn.Parameter, torch.nn.Conv1d, torch.nn.Conv2d, MaskedConv1D, torch.nn.Conv3d, torch.nn.MultiheadAttention, torch.nn.ConvTranspose1d)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm,torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Embedding, torch.nn.LayerNorm)

    # three sets of lr: low_lr for feat_extractor/i3d, med_lr for neck/backbone, and high_lr for heads. 
    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            #low lr for feat extractor
            if "feat_extractor" in fpn or "feat_extractor" in pn:
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    low_lr_no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    low_lr_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    low_lr_no_decay.add(fpn)
                elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    low_lr_no_decay.add(fpn)
                elif pn.endswith('rel_pe'):
                    # corner case for relative position encoding
                    low_lr_no_decay.add(fpn)
            # med lr for neck
            elif ("neck" in fpn or "neck" in pn) or ("backbone" in fpn or "backbone" in pn):
                if pn.endswith('bias'):
                    med_lr_no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    med_lr_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    med_lr_no_decay.add(fpn)
                elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                    med_lr_no_decay.add(fpn)
                elif pn.endswith('rel_pe'):
                    med_lr_no_decay.add(fpn)
            # high lr for quality_score_head
            elif ("quality_score_head" in fpn or "quality_score_head" in pn):
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    high_lr_no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    high_lr_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    high_lr_no_decay.add(fpn)
                elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    high_lr_no_decay.add(fpn)
                elif pn.endswith('rel_pe'):
                    # corner case for relative position encoding
                    high_lr_no_decay.add(fpn)
                elif pn.endswith('temperature'):
                    high_lr_no_decay.add(fpn)
            else:
                raise ValueError(f"Unrecognized parameter: {fpn}")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # @TODO add assert statements for med_lr_decay, med_lr_no_decay?
    #check that all intersections are empty
    assert len(high_lr_decay & high_lr_no_decay) == 0, \
        "parameters %s were in both high_lr_decay and high_lr_no_decay set!" \
        % (str(high_lr_decay & high_lr_no_decay), )
    assert len(low_lr_decay & low_lr_no_decay) == 0, \
        "parameters %s were in both low_lr_decay and low_lr_no_decay set!" \
        % (str(low_lr_decay & low_lr_no_decay), )
    assert len(high_lr_decay & low_lr_decay) == 0, \
        "parameters %s were in both high_lr_decay and low_lr_decay set!" \
        % (str(high_lr_decay & low_lr_decay), )
    assert len(high_lr_no_decay & low_lr_no_decay) == 0, \
        "parameters %s were in both high_lr_no_decay and low_lr_no_decay set!" \
        % (str(high_lr_no_decay & low_lr_no_decay), )
    

    union_params = high_lr_decay | high_lr_no_decay | low_lr_decay | low_lr_no_decay | med_lr_decay | med_lr_no_decay
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )
    
    # create the pytorch optimizer object
    if (len(low_lr_decay) + len(low_lr_no_decay) + len(med_lr_decay) + len(med_lr_no_decay)) > 0:
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(high_lr_decay))],
                        "weight_decay": optimizer_config['weight_decay'],
                        'name': 'high_lr_decay'},
            {"params": [param_dict[pn] for pn in sorted(list(high_lr_no_decay))],
                       "weight_decay": 0.0,
                       'name': 'high_lr_no_decay'},
            {"params": [param_dict[pn] for pn in sorted(list(med_lr_decay))],
                        "weight_decay": optimizer_config['weight_decay'],
                        "lr": optimizer_config["learning_rate"] * optimizer_config["neck_lr_factor"],
                        'name': 'med_lr_decay'},
            {"params": [param_dict[pn] for pn in sorted(list(med_lr_no_decay))],
                        "weight_decay": 0.0,
                        "lr": optimizer_config["learning_rate"] * optimizer_config["neck_lr_factor"],
                        'name': 'med_lr_no_decay'},
            {"params": [param_dict[pn] for pn in sorted(list(low_lr_decay))],
                        "weight_decay": optimizer_config['weight_decay'],
                        "lr": optimizer_config["learning_rate"] * optimizer_config["feature_extractor_factor"],
                        'name': 'low_lr_decay'},
            {"params": [param_dict[pn] for pn in sorted(list(low_lr_no_decay))],
                        "weight_decay": 0.0,
                        "lr": optimizer_config["learning_rate"] * optimizer_config["feature_extractor_factor"],
                        'name': 'low_lr_no_decay'}
        ]
    else:
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(high_lr_decay))], "weight_decay": optimizer_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(high_lr_no_decay))], "weight_decay": 0.0}
        ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        elif optimizer_config["schedule_type"] == "no_decay":
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupNoDecayLR(
                optimizer,
                warmup_steps,
                steps,
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def my_corn_label_from_logits(logits):
    probs =  torch.sigmoid(logits)
    cum_prob = torch.cumprod(probs, dim=-1)
    predict_levels = cum_prob > 0.5
    predict_labels = torch.sum(predict_levels, dim=-1)
    return predict_labels



def print_training_progress(curr_epoch, curr_iter, total_iters, batch_time, losses_tracker, rho, L2, RL2):
    # Format the training progress information
    progress_str = [
        f"Epoch: [{curr_epoch:03d}][{curr_iter:05d}/{total_iters:05d}]",
        f"Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s)\n",
        f"Total Loss: {losses_tracker['final_loss'].val:.2f} ({losses_tracker['final_loss'].avg:.2f})"
    ]
    
    loss_info = ", ".join([f"{key}: {value.val:.2f} ({value.avg:.2f})" for key, value in losses_tracker.items() if key != "final_loss"])
    progress_str.append(loss_info)
    
    progress_str.append(f"Running -> Rho: {rho:.4f}, L2: {L2:.2f}, RL2: {RL2:.4f}")
    
    print('\t'.join(progress_str))




################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    cfg = None,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = defaultdict(AverageMeter)
    metric_dict = MetricsDict()
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()
    
    #fix bn
    model.apply(set_bn_eval)

    if "freeze_early_layers" in cfg["train_cfg"]:
        if cfg["train_cfg"]["freeze_early_layers"]:
            print("Freezing early layers")
            model.apply(freeze_top_3_blocks)

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()

    optimizer.zero_grad()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optimizer
        preprocessed_dict = preprocessing(video_list,
                                    feat_extractor_type = cfg["model"]["feat_extractor_type"],
                                    max_seq_len= cfg["model"]["max_seq_len"],
                                    num_phases= cfg["model"]["num_phases"]
                                    )       

        #loss is calculated in the model
        criterion_args = {"target" : preprocessed_dict['target'],
                "difficulties" : preprocessed_dict["difficulties"],
                "gt_actions" : preprocessed_dict["gt_actions"],
                "loss_weights" : cfg["train_cfg"]["loss_weights"],
                "with_dd" : cfg["dataset"]["with_dd"],
                "three_judge_score_scaling" : cfg["dataset"]["three_judge_score_scaling"]
            }                                                
        # forward
        model_output, losses = model(**dict(batched_inputs = preprocessed_dict["batched_inputs"],
                            batched_masks = preprocessed_dict["batched_masks"],
                            batched_gt_actions = preprocessed_dict["gt_actions"],
                            video_ids = preprocessed_dict["video_ids"], 
                            curr_epoch = curr_epoch,
                            criterion_args = criterion_args
                            ))


        for key, value in losses.items():
            losses[key]  = losses[key].mean()
        
        losses["final_loss"] /= cfg["train_cfg"]["accumulation_steps"]

        losses['final_loss'].backward()

        # gradient cliping (to stabilize training if necessary)
        if cfg["train_cfg"]["clip_grad_l2norm"] > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["train_cfg"]["clip_grad_l2norm"]
            )
        
        if ((iter_idx + 1) % cfg["train_cfg"]["accumulation_steps"]) == 0 or (iter_idx == num_iters - 1):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


        with torch.no_grad():
            update_metric_dict_with_model_output(metric_dict, model_output, preprocessed_dict['gt_scores'],preprocessed_dict['difficulties'], is_val=False, cfg=cfg)

            # track all losses
            for key, value in losses.items():
                losses_tracker[key].update(value.item())
            
            del losses

            gc.collect()

            # printing (only check the stats when necessary to avoid extra cost)
            if ((iter_idx != 0) and (iter_idx % print_freq) == 0) or (iter_idx == num_iters - 1):
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / print_freq)
                start = time.time()

                rho, L2, RL2 = evaluate(metric_dict,
                                        is_train=True,
                                        dataset_name=cfg["dataset_name"],
                                        )

                # log to tensor board
                lr = scheduler.get_last_lr()[0]
                global_step = curr_epoch * num_iters + iter_idx
                if tb_writer is not None:
                    # learning rate (after stepping)
                    tb_writer.add_scalar(
                        'train/learning_rate',
                        lr,
                        global_step
                    )
                    # all losses
                    losses_vals_dict = {key : value.val for key, value in losses_tracker.items()}
                    
                    tb_writer.add_scalars('train/all_losses', losses_vals_dict, global_step)
                    tb_writer.add_scalars('train/L2, RL2', {'L2': L2, 'RL2': RL2}, global_step)
                    tb_writer.add_scalar('train/rho', rho, global_step)

                print_training_progress(curr_epoch, iter_idx, num_iters, batch_time, losses_tracker, rho, L2, RL2)

    rho, L2, RL2 = evaluate(metric_dict,
                            is_train = True,
                            dataset_name = cfg["dataset_name"]
                            )

    
    print('Full epoch metrics -> Rho : {:.4f}, L2 : {:.2f}, RL2 : {:.4f}'.format(rho, L2, RL2))

    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))

    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    cfg,
    tb_writer,
    print_freq,
    save_predictions = False
):
    
    """Test the model on the validation set"""

    # switch to evaluate mode
    model.eval()

    num_iters = len(val_loader)
    metric_dict = MetricsDict()

    # loop over validation set
    print(["Validation"])
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            preprocessed_dict = preprocessing(video_list,
                                        feat_extractor_type = cfg["model"]["feat_extractor_type"],
                                        max_seq_len= cfg["model"]["max_seq_len"],
                                        num_phases= cfg["model"]["num_phases"]
                                        )                                                                           
            # forward / backward the model
            model_output, _ = model(**dict(batched_inputs = preprocessed_dict["batched_inputs"],
                                    batched_masks = preprocessed_dict["batched_masks"],
                                    batched_gt_actions = preprocessed_dict["gt_actions"],
                                    video_ids = preprocessed_dict["video_ids"], 
                                    curr_epoch = curr_epoch
                                    ))
            # only used for evaluation
            metric_dict.update("video_ids", preprocessed_dict["video_ids"])

            update_metric_dict_with_model_output(metric_dict, model_output, preprocessed_dict['gt_scores'],preprocessed_dict['difficulties'], is_val=True, cfg=cfg)

        if ((iter_idx != 0) and (iter_idx % print_freq) == 0) or (iter_idx == num_iters - 1):
            torch.cuda.synchronize()

            block0 = f'Epoch: [{curr_epoch:03d}][{iter_idx:05d}/{num_iters:05d}]'
            print(block0)


    rho, L2, RL2 = evaluate(metric_dict,
                                        is_train=False,
                                        dataset_name = cfg["dataset_name"],
                                        curr_epoch = curr_epoch
                                        )
  
    print('Eval Metrics -> Rho : {:.4f}, L2 : {:.2f}, RL2 : {:.4f}'.format(rho, L2, RL2))

    # log metrics to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/rho', rho, curr_epoch)
        tb_writer.add_scalars('validation/L2, RL2', {'L2': L2, 'RL2': RL2}, curr_epoch)
    
    if save_predictions:
        metric_dict = metric_dict.get_metric_dict()
    else:
        metric_dict = None

    return rho, RL2, metric_dict