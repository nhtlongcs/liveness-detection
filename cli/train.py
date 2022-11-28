import torch
from core.opt import Opts

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import seed_everything

from core.models import MODEL_REGISTRY
from core_pkg.models import MODEL_REGISTRY
from core.callbacks import CALLBACKS_REGISTRY
from core.utils.path import prepare_checkpoint_path
import tabulate


def train(config):
    model = MODEL_REGISTRY.get(config["model"]["name"])(config)
    pretrained_path = config["global"]["pretrained"]

    if pretrained_path:
        model = model.load_from_checkpoint(pretrained_path, config=config)

    cp_path, train_id = prepare_checkpoint_path(config["global"]["save_dir"],
                                                config["global"]["name"])

    callbacks = [
        CALLBACKS_REGISTRY.get(mcfg["name"])(**mcfg["args"])
        for mcfg in config["callbacks"]
    ]

    Wlogger = WandbLogger(
        project=config["global"]["project_name"],
        name=train_id,
        save_dir=config["global"]["save_dir"],
        # log_model="all",
        entity=config["global"]["username"],
    )

    # Save config to wandb

    trainer = pl.Trainer(
        default_root_dir=cp_path,
        max_epochs=config.trainer["num_epochs"],
        gpus=-1
        if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=config.trainer["evaluate_interval"],
        log_every_n_steps=config.trainer["print_interval"],
        enable_checkpointing=True,
        strategy="ddp" if (torch.cuda.device_count() > 1
                           and not (config['global']['find_lr'])) else None,
        sync_batchnorm=True if
        (torch.cuda.device_count() > 1
         and not (config['global']['find_lr'])) else False,
        precision=16 if config["global"]["use_fp16"] else 32,
        logger=Wlogger,
        callbacks=callbacks,
        num_sanity_val_steps=20,
        deterministic=True,
        replace_sampler_ddp=False,  # This is important for two stream
        accumulate_grad_batches=config.trainer["accumulate_grad_batches"],

        # num_sanity_val_steps=-1,  # Sanity full validation required for visualization callbacks
    )

    if trainer.global_rank == 0:
        Wlogger.experiment.config.update(config)

    if config['global']['find_lr']:
        print("You are using find_lr mode, the model will not be trained")
        lr_finder = trainer.tuner.lr_find(model)
        model.learning_rate = lr_finder.suggestion()
        table = tabulate.tabulate([['learning_rate', model.learning_rate]],
                                  headers=['name', 'value'],
                                  tablefmt='fancy_grid')
        print(table)
    else:
        print(
            "If this is the first time you run this model, you can use global.find_lr=True to find the best lr"
        )
        trainer.fit(model, ckpt_path=config["global"]["resume"])


if __name__ == "__main__":
    cfg = Opts().parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    train(cfg)
