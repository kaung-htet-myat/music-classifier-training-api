import wandb

def wandb_init(wandb_cfg, data_cfg, model_cfg, training_cfg, exp_name):

    configs = {
        "data": dict(data_cfg),
        "model": dict(model_cfg),
        "training": dict(training_cfg)
    }

    wandb.init(
        project=wandb_cfg.project_name,
        entity=wandb_cfg.entity,
        config=configs,
        group=wandb_cfg.group if wandb_cfg.group else None,
        resume=wandb_cfg.resume,
        name=wandb_cfg.name if wandb_cfg.name else exp_name,
        id=wandb_cfg.id if wandb_cfg.id else exp_name,
    )

    return wandb