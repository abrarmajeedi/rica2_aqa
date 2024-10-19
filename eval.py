# python imports
import argparse
import os
from pprint import pprint
import sys, pickle

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (Logger, valid_one_epoch, 
                        fix_random_seed, ModelEma)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def override_cfg_params(cfg, args):
    if args.gpu is not None:
        cfg["devices"] = args.gpu
    else:
        cfg['devices'] = [i for i in range(torch.cuda.device_count())]
        
    if args.data_root is not None:
        cfg["dataset"]["data_root"] = args.data_root

    if args.test_batch_size > 0:
        cfg["loader"]["test_batch_size"] = args.test_batch_size

    if cfg["train_cfg"]["loss_weights"]["loss"] == "corn":
        cfg["model"]["score_bins"] -= 1

    if args.cv_fold > -1:
        cfg["dataset"]["cross_val_id"] = args.cv_fold

    if args.cv_split_file != "":
        cfg["dataset"]["cross_val_split_file"] = args.cv_split_file

    if cfg["dataset"]["use_feats"] == False:
        cfg["model"]["finetune_feat_extractor"]= True
        cfg["model"]["feat_extractor_type"]= 'i3d'
        cfg["model"]["feat_extractor_weights_path"]= './pre_trained/model_rgb.pth'

    if cfg["model"]["use_stochastic_embd"] == False:
        cfg["train_cfg"]["loss_weights"]["phase_vib"] = 0.0
        
    return cfg



def create_train_val_dataloaders(cfg, rng_generator):
    train_dataset = make_dataset(
        cfg['dataset_name'],
        True,
        cfg['train_split'],
        **cfg['dataset']
    )


    val_dataset = make_dataset(
        cfg['dataset_name'],
        False,
        cfg['val_split'],
        **cfg['dataset']
    )

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    val_loader = make_data_loader(
        val_dataset, False, None, **cfg['loader'])
    
    return (train_loader, val_loader)

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
   
    # parse args
    args.start_epoch = 0

    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    torch.set_warn_always(False)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    cfg = override_cfg_params(cfg, args)
    
    print("Args")
    pprint(vars(args), indent=4, stream=sys.__stdout__,sort_dicts=False)
    pprint(cfg, stream=sys.__stdout__,sort_dicts=False)
    
    """2. create dataset / dataloader"""
    train_loader, val_loader = create_train_val_dataloaders(cfg, rng_generator)

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    # not ideal for multi GPU training, ok for now
    # gpu_ids = ','.join(str(device_id) for device_id in cfg['devices'])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # model = nn.DataParallel(model, device_ids=cfg['devices'])
    model = nn.DataParallel(model).cuda()

    ckpt_file = args.ckpt

    if not os.path.isfile(ckpt_file):
        raise ValueError("CKPT file does not exist!")

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)   
    del checkpoint


    """5. validation loop"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))

    with torch.no_grad():
        curr_srcc, curr_rl2, metric_dict = valid_one_epoch(
            val_loader,
            model,
            -1,
            cfg = cfg,
            tb_writer=None,
            print_freq=args.print_freq,
            save_predictions=True
        )

    print("SRCC: {:.4f}, RL2: {:.4f}".format(curr_srcc, curr_rl2)) 

    with open(os.path.join(os.path.dirname(ckpt_file), "epoch_{:03d}_srcc_{:.3f}_rl2_{:.3f}_outputs.pkl".format(-1, curr_srcc, curr_rl2)), "wb") as f:
        pickle.dump(metric_dict, f)


    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--ckpt', default='', type=str,
                        help='name of exp folder (default: none)')   
    parser.add_argument('--data_root', type=str, metavar='PATH',)
    parser.add_argument('--test_batch_size', default=-1, type=int)
    parser.add_argument('--cv_fold', default=-1, type=int)
    
    parser.add_argument('--cv_split_file', default='', type=str)
    parser.add_argument('--gpu', nargs='*')

    args = parser.parse_args()

    main(args)
