import logging
import os
import json
import numpy as np

# from tensorflow.python.summary import event_accumulator
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def get_metrics(event_file, *scalar_str):
    # thank you https://sakishinoda.github.io/2017/02/13/tensorflow-summary-event-files-as-numpy-arrays.html
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    steps = {}
    values = {}
    scalars = ea.Tags()["scalars"]
    not_found = []
    for s in scalar_str:
        if not s in scalars:
            not_found.append(s)
            continue
        steps[s] = []
        values[s] = []
        for scalar in ea.Scalars(s):
            steps[s].append(scalar.step)
            values[s].append(scalar.value)
    if len(not_found) > 0:
        print(f"Did not find {not_found} in file {event_file}, found {scalars}")
    steps = {k: np.array(v) for k, v in steps.items()}
    values = {k: np.array(v) for k, v in values.items()}
    if len(scalar_str) == 1 and len(steps) >= 1:
        steps = steps[scalar_str[0]]
        values = values[scalar_str[0]]
    return steps, values


def get_runs(run_dir):
    """

    :param run_dir: sacred logdir from the experiment
    :return: config of the run, run path, event file path
    """
    run_dir = os.path.abspath(run_dir)
    outs = []
    dirs = os.listdir(run_dir)
    logging.debug(dirs)
    for r in dirs:
        rp = os.path.join(run_dir, r)
        if "_sources" in rp or not os.path.isdir(rp):
            continue
        with open(os.path.join(rp, "config.json")) as f:
            c = json.load(f)
        for dp, dn, fn in os.walk(rp):
            ep = None
            for f in fn:
                if "events" in f:
                    ep = os.path.join(dp, f)
                    outs.append((c, rp, ep))
                    break
            if ep is not None:
                break

    return outs
