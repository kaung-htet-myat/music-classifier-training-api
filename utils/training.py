import torch


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def get_optimizer(model, optimizer_cfg):
    optimizer = None

    if optimizer_cfg.method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg.lr)
    return optimizer


def get_scheduler(optimizer, scheduler_cfg):
    lr_scheduler = None

    if scheduler_cfg.method == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
    return lr_scheduler


def train_epoch(device, epoch, model, dataloader, loss_func, optimizer):
    print(f"epoch {epoch}")
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
        # xm.mark_step()

        running_loss = loss.item()
        train_loss += running_loss

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {running_loss} [{current:>5d}]")

    train_loss = train_loss / batches
    print(f"train loss: {train_loss}\tlr: {get_lr(optimizer)}")

    # torch.save(model.state_dict(), CHECKPOINT_PATH+f'epoch_{epoch}_weights.pth')
    # xm.save(model.state_dict(), CHECKPOINT_PATH+f'epoch_{epoch}_weights.pth') # for xla


def test_epoch(device, model, dataloader, loss_func):
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

    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {loss:>8f}")
    print("="*20)