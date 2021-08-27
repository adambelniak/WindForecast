def transformer_lr_scheduler(epoch: int, warmup_epochs: int, decay_epochs: int, initial_lr: float, base_lr: float, final_lr: float):
    if epoch < warmup_epochs:
        diff = base_lr - initial_lr
        rate = (epoch + 1) / warmup_epochs
        return (diff * rate) / base_lr

    if warmup_epochs <= epoch < warmup_epochs + decay_epochs:
        diff = base_lr - final_lr
        rate = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return (diff * rate) / base_lr

    return final_lr / base_lr
