import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class Save_results(Callback):
    def __init__(self):
        super().__init__()
        pass

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """
        Run the save process at the end of the training
        """
        epoch = trainer.current_epoch
        print("Epoch current", epoch)