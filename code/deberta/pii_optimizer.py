# import bitsandbytes as bnb
from torch import optim

"""
Customizing the optimizer: lr for head_layers vs non_head layers, no weight decay for bias & LayerNorm 
"""

def get_optimizer(cfg, model, print_fn=None):
    # Supported optimizers
    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        # "AdamW8bit": bnb.optim.Adam8bit,
    }

    optimizer_name = cfg.optimizer.name
    if optimizer_name not in _optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    optimizer_class = _optimizers[optimizer_name]

    no_decay = {"bias", "LayerNorm.weight"}
    head_layer_names = ["classification_head"]

    optim_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  

        is_head_param = any(head_layer_name in name for head_layer_name in head_layer_names)

        lr = cfg.optimizer.head_lr if is_head_param else cfg.optimizer.lr
        weight_decay = 0 if any(nd in name for nd in no_decay) else cfg.optimizer.weight_decay

        optim_groups.append({'params': [param], 'lr': lr, 'weight_decay': weight_decay})

    # if print_fn is not None:
    #     for idx, group in enumerate(optim_groups):
    #         n_params = round(sum(p.numel() for p in group['params'])/1e6, 2)
    #         print_fn(f"Group {idx}: LR={group['lr']}, Weight Decay={group['weight_decay']}, Params={n_params}M")

    optimizer = optimizer_class(optim_groups)
    return optimizer
