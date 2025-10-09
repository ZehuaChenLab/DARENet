import torch


class Saver(object):
    def __init__(self):
        self.checkpoint = None
        self.best_weight = None

    def save_checkpoint(self, solver, epoch, train_epoch_best_loss, total_time, no_optim, path_checkpoint):
        self.checkpoint = {"model_state_dict": solver.net.state_dict(),
                           'scheduler_state_dict': solver.scheduler.state_dict(),
                           "optimizer_state_dict": solver.optimizer.state_dict(),
                           "epoch": epoch,
                           "train_epoch_best_loss": train_epoch_best_loss,
                           "total_time": total_time,
                           "no_optim": no_optim}
        torch.save(self.checkpoint, path_checkpoint)

    def load_checkpoint(self, solver):
        solver.load_state_dict(self.checkpoint['model_state_dict'])
        solver.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        solver.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

    def save_best_weight(self, path):
        torch.save(self.best_weight, path)

