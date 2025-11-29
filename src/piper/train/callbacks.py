import os
from typing import Optional

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class LastCheckpoint(Callback):
    def __init__(
        self,
        every_n_epochs: Optional[int] = None,
        dirpath: Optional[str] = None,
        filename: str = "last.ckpt",
        save_on_train_end: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.dirpath = dirpath
        self.filename = filename
        self.save_on_train_end = save_on_train_end
        self.verbose = verbose

    def _get_checkpoint_path(self, trainer) -> str:
        if self.dirpath is not None:
            base_dir = self.dirpath
        else:
            log_dir = getattr(trainer, "log_dir", None)
            if log_dir is None:
                log_dir = trainer.default_root_dir
            base_dir = os.path.join(log_dir, "checkpoints")

        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, self.filename)

    def _save_last(self, trainer) -> None:
        path = self._get_checkpoint_path(trainer)
        if self.verbose:
            rank_zero_info(f"[LastCheckpoint] Saving last checkpoint to {path}")
        trainer.save_checkpoint(path, weights_only=False)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs is None or self.every_n_epochs <= 0:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0:
            self._save_last(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        if self.save_on_train_end:
            self._save_last(trainer)