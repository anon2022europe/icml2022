# -*- coding: utf-8 -*-

from ggg.experiments.ggan_exp import *
import sys
# some hackery to make calling this from a script easier
_THIS_FILE_PATH=os.path.abspath(__file__)

if __name__ == "__main__":
    ex.run_commandline(sys.argv)
