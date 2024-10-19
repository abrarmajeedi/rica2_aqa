import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "",
    "devices": ['cuda:0'], # default: single gpu
    "train_split": ('training', ),
    "val_split": ('validation', ),
    "model_name": "XXX",
    "dataset": {

    },
    "loader": {
        "train_batch_size": 8,
        "test_batch_size": 8,
        "num_workers": 1,
    },
    # network architecture
    "model": {
        "finetune_feat_extractor": False,
        "feat_extractor_type": None,
        "feat_extractor_weights_path": None,
        "backbone_type": 'xxx',
        # disable abs position encoding (added to input embedding)
        "neck_type": 'xxx',
        "decoder_params": {
                "decoder_ffn_dim": 2048,
                "decoder_activation": 'gelu',
        },
    },
    "train_cfg": {
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
    },
    "test_cfg": {
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # fill in derived fields
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    config["model"]["frames_per_clip"] = config["dataset"]["frames_per_clip"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config