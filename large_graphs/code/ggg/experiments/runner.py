import sys
import os
import argparse
import pathlib

import subprocess


def hpc_wrap(cmd, enable_gpu=True):
    """Takes a python script and wraps it in `sbatch` over `ssh`.

    :param cmd: The python script to be executed.
    :param enable_gpu:
    :return: Return array that can be executed with `subprocess.call`.
    """
    python_cmd_args = " ".join(map(lambda x: "'{}'".format(x), cmd))

    if enable_gpu:
        bash_script = "hpc.sh"
    else:
        bash_script = "hpc-cpu.sh"

    server_cmd = "cd graph-gan-main/code; sbatch {} {}".format(
        bash_script, python_cmd_args
    )
    ssh_cmd = ["ssh", "anon", server_cmd]
    return ssh_cmd


def server_execute(cmd, enable_gpu=True):
    """Executes a script over `ssh` using the SLURM queuing system.

    :param cmd:
    :param enable_gpu:
    :return:
    """
    ssh_cmd = hpc_wrap(cmd, enable_gpu=enable_gpu)
    print(ssh_cmd)
    print(subprocess.check_output(ssh_cmd))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention runs")
    parser.add_argument("--run_dir", type=str, default="ggg/experiments/")
    parser.add_argument("--exp_name", type=str, default="GGG.py")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--remote", action="store_true", default=False, help="execute on HPC."
    )
    parser.add_argument(
        "--dummy-run",
        action="store_true",
        default=False,
        help="Run without doing anything.",
    )
    args = parser.parse_args()

    cmd_list = sys.argv

    if args.config is None:
        configs_name = ["report_base_comm100_RF1",
                        "report_base_comm100_RF2", "report_base_comm100_RF3",
                        "report_base_zync_RF1", "report_base_zync_RF2",
                        "report_base_zync_RF3", "report_base_zync_RF4",
                        "report_base_zync_RF5",
                        "report_base_ogb_mol_RF1", "report_base_ogb_mol_RF2",
                        "report_base_ogb_mol_RF3", "report_base_ogb_mol_RF4",
                        # "report_base_mnist_RF1", "report_base_mnist_RF1",
                        #"report_base_chordal9_RF1", #"report_base_chordal9_RF2", "report_base_chordal9_RF3",
                        #"report_base_chordal9_RS1", "report_base_chordal9_RS2", "report_base_chordal9_RS3",
                        ]
    else:
        configs_name = [args.config]

    for config_n in configs_name:
        args.config = config_n
        if args.remote:
            while "--remote" in cmd_list:
                cmd_list.remove("--remote")
            server_execute(
                ["python"] + [args.run_dir + args.exp_name] + ["with"] + [args.config], enable_gpu=args.cuda
            )

        elif args.dummy_run:
            pass

        else:
            pass
