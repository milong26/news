"""
仿照lerobot_train写的带有validation功能的train
"""

#!/usr/bin/env python
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging
from torch.utils.data import Subset
from typing import Tuple
import random

import torch.nn.functional as F  # noqa: N812
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# 划分数据集，按 episode 划分
def split_dataset(dataset, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Subset, Subset]:
    """
    Split an episodic dataset into train and validation sets by episode.

    Args:
        dataset: Dataset object, expected to have `meta.episodes` dict with keys 'dataset_from_index' and 'dataset_to_index'
        train_ratio: Fraction of episodes to use for training (0 < train_ratio <= 1)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset: two Subset instances
    """
    if not hasattr(dataset, "meta") or not hasattr(dataset.meta, "episodes"):
        raise ValueError("Dataset must have meta.episodes for episodic splitting")

    episodes_meta = dataset.meta.episodes  # dict with keys: dataset_from_index, dataset_to_index
    num_episodes = len(episodes_meta["dataset_from_index"])
    indices = list(range(num_episodes))
    random.seed(seed)
    random.shuffle(indices)

    split_idx = int(train_ratio * num_episodes)
    train_episode_indices = indices[:split_idx]
    val_episode_indices = indices[split_idx:]

    # Convert episode indices to frame indices
    def episodes_to_frame_indices(episode_indices):
        frame_indices = []
        for idx in episode_indices:
            start = episodes_meta["dataset_from_index"][idx]
            end = episodes_meta["dataset_to_index"][idx]
            frame_indices.extend(range(start, end))
        return frame_indices

    train_frame_indices = episodes_to_frame_indices(train_episode_indices)
    val_frame_indices = episodes_to_frame_indices(val_episode_indices)

    # Store episode info for sampler if needed
    dataset.episodes = {
        "train": train_episode_indices,
        "val": val_episode_indices
    }

    train_dataset = Subset(dataset, train_frame_indices)
    val_dataset = Subset(dataset, val_frame_indices)
    return train_dataset, val_dataset

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    with accelerator.autocast():
        if rabc_batch_weights is not None:
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)
    accelerator.backward(loss)
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)

    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train_and_validate(cfg: TrainPipelineConfig, train_ratio: float = 0.8, accelerator: Accelerator | None = None):
    """Train and validate with adjustable train/val split ratio."""
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    cfg.validate()
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        train_dataset, val_dataset = split_dataset(dataset, train_ratio)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(cfg)
        train_dataset, val_dataset = split_dataset(dataset, train_ratio)

    # Policy
    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    accelerator.wait_for_everyone()

    # Processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path, **processor_kwargs, **postprocessor_kwargs)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=False)

    # Prepare with accelerator
    policy, optimizer, train_loader, lr_scheduler = accelerator.prepare(policy, optimizer, train_loader, lr_scheduler)
    val_loader = accelerator.prepare(val_loader)
    dl_iter = cycle(train_loader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(effective_batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=0, accelerator=accelerator)

    for step in range(cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator=accelerator, lr_scheduler=lr_scheduler
        )
        train_tracker.step()

        # Logging
        if cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)

        # Validation验证集评估
        if cfg.eval_freq > 0 and step % cfg.eval_freq == 0 and is_main_process:
            policy.eval()
            policy.model.eval()
            val_metrics = {"loss": AverageMeter("val_loss", ":.3f")}
            val_tracker = MetricsTracker(effective_batch_size, len(val_dataset), 1, val_metrics, initial_step=step, accelerator=accelerator)

            with torch.no_grad(), accelerator.autocast():
                for val_batch in val_loader:
                    val_batch = preprocessor(val_batch)

                    # 直接预测动作，不调用 forward 避免 KL 计算
                    actions_hat = policy.predict_action_chunk(val_batch)

                    # 手动计算 L1 loss
                    l1_loss = (
                        F.l1_loss(val_batch[ACTION], actions_hat, reduction="none")
                        * ~val_batch["action_is_pad"].unsqueeze(-1)
                    ).mean()

                    val_tracker.loss = l1_loss.item()
                    val_tracker.step()

            if wandb_logger:
                wandb_logger.log_dict(val_tracker.to_dict(), step, mode="eval")

            policy.model.train()
            policy.train()


def main():
    register_third_party_plugins()
    train_and_validate()


if __name__ == "__main__":
    main()
