class TransformerLRScheduler:

    def __init__(self, warmup_epochs: int, decay_epochs: int, starting_lr: float, base_lr: float, final_lr: float):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.starting_lr = starting_lr
        self.base_lr = base_lr
        self.final_lr = final_lr

    def transformer_lr_scheduler(self, epoch: int):
        if epoch < self.warmup_epochs:
            diff = self.base_lr - self.starting_lr
            if diff == 0:
                return self.base_lr
            rate = (epoch + 1) / self.warmup_epochs
            return (diff * rate) / self.base_lr

        if self.warmup_epochs <= epoch < self.warmup_epochs + self.decay_epochs:
            diff = self.base_lr - self.final_lr
            if diff == 0:
                return self.base_lr
            rate = 1 - ((epoch - self.warmup_epochs) / self.decay_epochs)
            return (diff * rate) / self.base_lr

        return self.final_lr / self.base_lr
