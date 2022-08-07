import json
import os
import argparse
import torch

import sys 
sys.path.append('..')

from utils.main_utils import add_args, get_net_params, get_parameters


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def script_boilerplate(args):
    num_cards = len(args.param_values)
    model = args.model
    dataset = args.dataset
    job_note = args.job_note
    return f"""#!/bin/bash
#$ -N {model}_{dataset}_{job_note}
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-{num_cards}:1

"""

def pre_run_boilerplate(args):
    debug_file = f"fname=$(pwd)/{args.job_note}_${{SGE_TASK_ID}}_${{args.varying_param[${{SGE_TASK_ID}}]}}_DEBUG.txt"
    rest = f"""touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

"""
    return debug_file + "\n" + rest


def main(args):
    if len(args.param_values) > 5:
        raise ValueError(f'Too many param values for {args.varying_param}')

    script_string = ""

    with open(args.config) as f:
        config = json.load(f)
    params = get_parameters(config, args)
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    args.model = config["model"]
    args.dataset = config["dataset"]

    if args.dataset is not None:
        DATASET_NAME = args.dataset

    net_params = get_net_params(config, args, device, params, DATASET_NAME)

    script_string += script_boilerplate(args)

    varying_param_str = f"{args.varying_param}=({' '.join(args.param_values)})"
    script_string += varying_param_str + "\n"
    script_string += pre_run_boilerplate(args)
    with open('./test.sh', 'w') as f:
        f.write(script_string)
    print(script_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--varying_param", type=str, help="Parameter to vary, only one allowed")
    parser.add_argument("--param_values", nargs='+', help="Values to vary, max 5 values")
    parser.add_argument("--job_note", type=str, help="Job note for job name")
    parser = add_args(parser)
    main(parser.parse_args())