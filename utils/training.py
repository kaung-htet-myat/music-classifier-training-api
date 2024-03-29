import os
import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_optimizer(model, optimizer_cfg):

    if optimizer_cfg.method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg.lr)

    return optimizer


def lrfn(epoch):
    if epoch < 5:
        lr = 1
    elif epoch < 15:
        lr = 0.5
    elif epoch < 25:
        lr = 0.1
    elif epoch < 35:
        lr = 0.05
    elif epoch < 40:
        lr = 0.01

    elif epoch < 45:
        lr = 1
    elif epoch < 55:
        lr = 0.5
    elif epoch < 65:
        lr = 0.1
    elif epoch < 75:
        lr = 0.05
    elif epoch < 80:
        lr = 0.01

    elif epoch < 85:
        lr = 0.5
    elif epoch < 95:
        lr = 0.1
    elif epoch < 100:
        lr = 0.05

    else:
        lr = 0.01
    return lr


def get_scheduler(optimizer, scheduler_cfg):

    if scheduler_cfg.method == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma
        )
    elif scheduler_cfg.method == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.t_0,
            T_mult=scheduler_cfg.t_mult,
            eta_min=scheduler_cfg.eta_min,
            last_epoch=scheduler_cfg.last_epoch,
        )
    elif scheduler_cfg.method == "lambda":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lrfn,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lrfn,
        )

    return lr_scheduler


def train_epoch(
    device,
    epoch,
    model,
    dataloader,
    loss_func,
    optimizer,
    scheduler,
    checkpoint_path,
    exp_name,
    wandb,
    logger,
):
    logger.info(f"epoch {epoch}:")
    batches = 0
    model.train().to(device)

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        batches += 1
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        train_loss += running_loss

        # if batch % 100 == 0:
        #     current = batch * len(X)
        #     print(f"loss: {running_loss} [{current:>5d}]")

    train_loss = train_loss / batches
    lr = get_lr(optimizer)
    logger.info(f"\ttrain loss: {train_loss}\tlr: {lr}")
    wandb.log(
        {
            "epoch": epoch,
            "loss": train_loss,
            "lr": lr,
        }
    )

    scheduler.step()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_loss,
        },
        os.path.join(checkpoint_path, f"{exp_name}_epoch_{epoch}.pth"),
    )


def test_epoch(device, epoch, model, dataloader, loss_func, logger, wandb=None):
    size = 0
    batches = 0
    model.eval().to(device)

    loss, acc = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            batches += 1
            size += X.size()[0]
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss += loss_func(pred, y).item()
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss = loss / batches
    acc = acc / size

    logger.info(f"\tval loss: {loss:>8f}\tval acc: {(100*acc):>0.1f}%")

    if wandb:
        wandb.log(
            {
                "epoch": epoch,
                "val_loss": loss,
                "val_acc": acc,
            }
        )
