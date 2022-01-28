from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from tensorboardX import SummaryWriter


def register_exp(exp: SummaryWriter):
    register_exp.writer = exp


def set_log_hists(lh):
    set_log_hists.log = lh


set_log_hists.log = None


def log_hists():
    return set_log_hists.log


def register_trainer(exp: Trainer):
    register_trainer.trainer = exp

def register_plm(plm):
    register_plm.model = plm

register_trainer.trainer = None
register_exp.writer = None
register_plm.model=None


def summarywriter() -> SummaryWriter:
    w = getattr(register_exp, "writer", None)
    if w is None:
        raise ValueError(
            "First call 'register_exp' with a summary writer in order to make this available"
        )
    return w


def global_step():
    return register_plm.model.global_step

def easy_hist(name,val,force=False):
    if log_hists() or force:
        summarywriter().add_histogram(name,val,global_step=global_step())
def tensor_imshow(name, X):
    if log_hists():
        X = X.detach().cpu().numpy()
        fig, ax = plt.subplots()
        mapp = ax.imshow(X)
        fig: plt.Figure
        fig.colorbar(mapp, ax=ax)
        summarywriter().add_figure(name, fig, global_step=global_step())
        plt.close(fig)
