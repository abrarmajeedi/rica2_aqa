# python imports
import argparse, os
import time
import datetime
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
from libs.utils import (Logger, train_one_epoch, valid_one_epoch, 
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed)

def override_cfg_params(cfg, args):
    if args.gpu is not None:
        cfg["devices"] = args.gpu
    else:
        cfg['devices'] = [i for i in range(torch.cuda.device_count())]
        
    if args.data_root is not None:
        cfg["dataset"]["data_root"] = args.data_root

    if args.train_batch_size > 0:
        cfg["loader"]["train_batch_size"] = args.train_batch_size

    if args.test_batch_size > 0:
        cfg["loader"]["test_batch_size"] = args.test_batch_size

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
        cfg["train_cfg"]["loss_weights"]["scale_vib"] = False

    return cfg


def create_checkpoint_folder(cfg, args):
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])

    cfg_filename = os.path.basename(args.config).replace('.yaml', '')

    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ts = str(ts).replace(' ', '_').replace(':', '-')
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(args.output))

    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    return ckpt_folder


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

    print("Train dataset size: {:d}".format(len(train_dataset)))
    print("Val dataset size: {:d}".format(len(val_dataset)))

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
    
    ckpt_folder = create_checkpoint_folder(cfg, args)
    print("Checkpoint folder: {:s}".format(ckpt_folder))
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    logger = Logger(os.path.join(ckpt_folder, '0_log.txt'))
    print("If you plan to debug using ipdb then comment the following line")
    #sys.stdout = logger

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

    model = nn.DataParallel(model).cuda()
    
    optimizer = make_optimizer(model, cfg['opt'])

    # schedule
    num_iters_per_epoch = len(train_loader) / cfg["train_cfg"]["accumulation_steps"]
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    best_rl2 = 1000
    srcc_at_best_rl2 = -100
    best_rl2_epoch = 0

    best_srcc = -100
    rl2_at_best_srcc = 1000
    best_srcc_epoch = 0

    for epoch in range(args.start_epoch, max_epochs):

        # train for one epoch
        #start_time = time.time()
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            cfg = cfg,
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        #print("Time taken for epoch: {:.2f} s".format(time.time() - start_time))

        if cfg["train_cfg"]["loss_weights"]["phase_vib"] > 0.0 and cfg["train_cfg"]["loss_weights"]["scale_vib"]:
            if epoch > 30 and epoch % 10 == 0:
                cfg["train_cfg"]["loss_weights"]["phase_vib"] *= 3
                cfg["train_cfg"]["loss_weights"]["phase_vib"] = min(cfg["train_cfg"]["loss_weights"]["phase_vib"], 0.005)
        

        if (epoch == 0) or ((epoch+1) <=  30 and (epoch+1) % 5 == 0 ) or (30 < (epoch+1) <= 120 and (epoch+1) % 3 == 0) or (epoch+1) > 120 or (epoch+1) == max_epochs:
            curr_srcc, curr_rl2, metric_dict = valid_one_epoch(
                val_loader,
                model,
                epoch,
                cfg = cfg,
                tb_writer=tb_writer,
                print_freq=args.print_freq,
                save_predictions=True
            ) 

            if curr_srcc > best_srcc:
                best_srcc = curr_srcc
                rl2_at_best_srcc = curr_rl2
                best_srcc_epoch = epoch
                srcc_improved = True
            else:
                srcc_improved = False
            
            if curr_rl2 < best_rl2:
                best_rl2 = curr_rl2
                srcc_at_best_rl2 = curr_srcc
                best_rl2_epoch = epoch
                rl2_improved = True
            else:
                rl2_improved = False

            print("Best SRCC: {:.4f}, corres. RL2: {:.4f} at epoch {:d}".format(best_srcc, rl2_at_best_srcc, best_srcc_epoch))
            print("Best RL2: {:.4f}, corres. SRCC: {:.4f} at epoch {:d}".format(best_rl2, srcc_at_best_rl2, best_rl2_epoch))

            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            if srcc_improved:
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='srcc_best.pth.tar'
                )
                with open(os.path.join(ckpt_folder, "srcc_best_outputs.pkl"), "wb") as f:
                    pickle.dump(metric_dict, f)

            if rl2_improved:
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='rl2_best.pth.tar'
                )
                with open(os.path.join(ckpt_folder, "rl2_best_outputs.pkl"), "wb") as f:
                    pickle.dump(metric_dict, f)

            del save_states

    print("Best SRCC: {:.4f}".format(best_srcc))
    print("Best RL2: {:.4f}".format(best_rl2))          

    # wrap up
    tb_writer.close()
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
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    
    parser.add_argument('--data_root', type=str, metavar='PATH',)
    parser.add_argument('--train_batch_size', default=-1, type=int)
    parser.add_argument('--test_batch_size', default=-1, type=int)
    parser.add_argument('--cv_fold', default=-1, type=int)
    
    parser.add_argument('--cv_split_file', default='', type=str)
    parser.add_argument('--gpu', nargs='*')

    args = parser.parse_args()

    main(args)
