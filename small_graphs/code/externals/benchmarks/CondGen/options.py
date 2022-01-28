import argparse

"""
epochs "converges to values of real  values with only around 100 epochs
DBLP: d=5,z_dim=6, lr=1e-3,epochs=100

TCGA: d=10,z_dim=6, lr=1e-3,epochs=100
"""
import attr


@attr.s
class AttrOptions:
    av_size = attr.ib(default=10)
    z_size = attr.ib(default=6)

    lr = attr.ib(default=1e-3)  # 3e-3 in original code
    beta = attr.ib(default=5)
    beta2 = attr.ib(default=0.1)
    alpha = attr.ib(default=0.1)
    gamma = attr.ib(default=15)
    gpu = attr.ib(default=0)
    DATA_DIR = attr.ib(default="data/dblp")
    output_dir = attr.ib(default="output")
    max_epochs = attr.ib(default=100)
    adj_thresh = attr.ib(default=0.6)
    gc_size = attr.ib(default=16)
    d_size = attr.ib(default=5)
    rep_size = attr.ib(default=32)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser("Training Parser")

    def initialize(self):
        parser = self.parser

        parser.add_argument(
            "--av_size", type=int, default=10, help="set 0 if you do not need attr_vec"
        )
        parser.add_argument("--z_size", type=int, default=6, help="noise vector")

        parser.add_argument("--rep_size", type=int, default=32, help="hidden vector")
        parser.add_argument("--d_size", type=int, default=5, help="d vector")
        parser.add_argument("--gc_size", type=int, default=16, help="gc vector")

        parser.add_argument(
            "--adj_thresh", type=float, default=0.6, help="threshold of adj edges"
        )
        parser.add_argument("--max_epochs", type=int, default=100, help="max epochs")

        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--beta", type=int, default=5, help="beta")
        parser.add_argument("--beta2", type=float, default=0.1, help="beta2")
        parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
        parser.add_argument("--gamma", type=int, default=15, help="gamma")

        parser.add_argument("--gpu", type=str, default="0", help="gpu id")
        parser.add_argument(
            "--DATA_DIR", type=str, default="./data/dblp/", help="output dir"
        )

        parser.add_argument(
            "--output_dir", type=str, default="./output/", help="output dir"
        )

        opt = parser.parse_args()
        return opt
