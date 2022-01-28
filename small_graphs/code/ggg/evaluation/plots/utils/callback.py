from tempfile import TemporaryDirectory
from typing import Optional
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from ggg.evaluation.plots.utils.post_experiment_plots import (
    generate_graphs,
    main_run_plot,
    main_run_MMD,
)
import os


class ModelCheckPointWithPlots(ModelCheckpoint):
    def __init__(
        self,
        filepath: Optional[str] = None,
        monitor: str = "val_loss",
        verbose: bool = False,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "auto",
        period: int = 50,
        prefix: str = "",
        numb_graphs=1024,
        plot_dataset=True,
        loss_dir=None,
        plot_dir=None,
        mmd=True,
        lcc=True,
    ):
        super().__init__(
            filepath,
            monitor,
            verbose,
            save_top_k,
            save_weights_only,
            mode,
            period=period,
            prefix=prefix,
        )

        if loss_dir is None:
            loss_dir = self.dirpath
        self.plot_dataset = plot_dataset
        self.numb_graphs = numb_graphs
        self.loss_dir = loss_dir
        self.plot_dir = plot_dir
        self.period = period
        self.mmd = mmd
        self.lcc = lcc

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # plot_dir=os.path.join(self.plot_dir,f"plots_epoch{trainer.current_epoch}")
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch > 0:
            os.makedirs(self.plot_dir, exist_ok=True)
            model = pl_module
            epoch = trainer.current_epoch
            data_dir = self.plot_dir
            plots_save_dir = self.plot_dir
            os.makedirs(data_dir, exist_ok=True)
            model.eval()
            gen_graphs, dataset_graphs = generate_graphs(
                model,
                current_epoch=trainer.current_epoch,
                device=model.device,
                batch_size=model.hpars.batch_size,
                dataset=model.train_set,
                numb_graphs=self.numb_graphs,
                save_dir=data_dir,
            )
            # TODO adapt to more than condgen: f"condgen_{model.hpars.dataset}"
            if self.verbose:
                print("Got graphs")
            plots = main_run_plot(
                trainer.current_epoch,
                model.hpars.exp_name,
                model.hpars.dataset_hpars.dataset,
                model_graphs=gen_graphs,
                dataset_graphs=dataset_graphs,
                loss_dir=None,
                plots_save_dir=plots_save_dir,
                lcc=self.lcc,
            )
            for k, v in plots.items():
                model.logger.experiment.add_figure(
                    k, v, global_step=trainer.current_epoch
                )
            if self.mmd:
                try:
                    main_run_MMD(
                        trainer.current_epoch,
                        csv_dir=plots_save_dir,
                        model_graphs=gen_graphs,
                        dataset_graphs=dataset_graphs,
                        numb_graphs=self.numb_graphs,
                    )
                except:
                    print("Error while trying ot calculate mmds")
            model.train()

        return super().on_validation_end(trainer, pl_module)
