from yacs.config import CfgNode as CN


_C = CN()
_C.MODEL = CN()
_C.TRAINER = CN()
_C.DATASET = CN()

def get_cfg_default():
    return _C.clone()


def clean_cfg(cfg, trainer):
    """Remove unused trainers (configs).
    Aim: Only show relevant information when calling print(cfg).
    Args:
        cfg (_C): cfg instance.
        trainer (str): trainer name.
    """
    keys = list(cfg.TRAINER.keys())
    for key in keys:
        if key == "NAME" or key == trainer.upper():
            continue
        cfg.TRAINER.pop(key, None)
        
        
def reset_cfg(cfg, args):
    if args.dataset:
        cfg.DATASET.NAME = args.dataset

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.test_domain:
        cfg.DATASET.TEST_DOMAIN = args.test_domain

    if args.algorithm:
        cfg.TRAINER.NAME = args.algorithm

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

