import logging
import math
import os
import random
import shutil
import string
from copy import deepcopy

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from transformers import get_cosine_schedule_with_warmup

logger = get_logger(__name__)


def generate_random_string(prefix="p_", num_chars=6):
    chars = string.ascii_lowercase + string.digits
    return prefix + ''.join(random.choice(chars) for _ in range(num_chars))


def get_desired_dtype(dtype):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32


def print_line(logger=None):
    prefix, unit, suffix = "#", "~~", "#"
    if logger is None:
        print(prefix + unit*50 + suffix)
    else:
        logger.print(prefix + unit*50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm%ds' % (m, s)


def init_wandb(cfg):
    project = cfg.wandb.project
    tags = cfg.wandb.tags

    if cfg.wandb.all_data_flag:
        run_id = f"{cfg.wandb.run_name}-all-data"
    else:
        run_id = f"{cfg.wandb.run_name}"

    run = wandb.init(
        project=project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )

    return run


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(cfg, state, is_best):
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    name = f"detect_ai_model"

    filename = f'{cfg.outputs.model_dir}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f'{cfg.outputs.model_dir}/{name}_best.pth.tar')


class EMA():
    """
    credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ---
def create_reference_df(examples):
    examples = deepcopy(examples)
    reference_df = pd.DataFrame(examples)[['document', 'tokens', 'labels']].copy()
    reference_df = reference_df.explode(['tokens', 'labels']).reset_index(
        drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
    reference_df['token'] = reference_df.groupby('document').cumcount()
    reference_df = reference_df[reference_df['label'] != 'O'].copy()

    # # ---
    # reference_df = reference_df[reference_df['label'].apply(lambda x: 'OTHER' not in x)].copy()
    # # ---

    reference_df = reference_df.reset_index().rename(columns={'index': 'row_id'})
    reference_df = reference_df[['row_id', 'document', 'token', 'label']].copy()
    return reference_df


def setup_training_run(cfg):
    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            log_with="wandb",
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.wandb.run_name}}
        )

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
        )

    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    return accelerator


def get_custom_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 10% of it,
    following a cosine curve, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of times the learning rate will decay to 10% of the maximum learning rate. Default: 0.5 (half a cycle).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Scale to decay to 10% of the max lr
        decay_target = 0.1  # Decay to 10% of the max lr
        decay_factor = (1 - decay_target) * cosine_decay + decay_target

        return decay_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# --
def reinit_deberta(base_model, num_reinit_layers):
    """re-initialize top n layers of the base deberta model

    :param base_model: base deberta model
    :type base_model: deberta model
    :param num_reinit_layers: number of top layers to re-initialize
    :type num_reinit_layers: int
    """
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
