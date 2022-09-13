import os
import argparse
import json

def get_script_text(job_name, v, command, idx="", seed=0):
    text = f"""#!/bin/bash
#$ -N {job_name}_{v}_{idx}_{seed}
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/{v}_DEBUG_{idx}_{seed}.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

{command}
"""
    return text

def main(args):
    file = args.file
    dir = os.path.dirname(file)

    # parse file into text format
    lines = open(file, 'r').readlines()

    job_name = lines[1].split(' ')[-1].strip()
    values = [int(x) for x in lines[6].split('=')[1].strip('()\n').split(' ')][1:]

    command = lines[14].strip().split(' ')

    job_num_idx = [i for i, x in enumerate(command) if 'job_num' in x][0] + 1
    pos_enc_dim_idx = [i for i, x in enumerate(command) if 'pos_enc_dim' in x][0] + 1
    test_json_idx = [i for i, x in enumerate(command) if '.json' in x][0]

    test_json_path = command[test_json_idx]
    test_json_path = '/'.join(test_json_path.split('/')[1:])
    test_config = json.load(open(test_json_path, 'rb'))
    seeds = test_config['params']['seed_array']

    for idx, value in enumerate(values):
        for seed in seeds:
            command[job_num_idx] = f'{value}'
            command[pos_enc_dim_idx] = f'{value}'
            command.append(f'--seed_array {seed}')
            with open(f'{dir}/{job_name}_{seed}_{idx}.sh', 'w') as f:
                f.write(get_script_text(job_name, value, ' '.join(command), idx=idx, seed=seed))
            command.pop(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to script to split")
    main(parser.parse_args())