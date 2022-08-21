#!/bin/bash
#$ -N GraphTransformer_ZINC_b64-lnorm-lap
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

pos_enc_dim=(0 4 8 16)
fname=$(pwd)/b64-lnorm-lap_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b64-lnorm-lap.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'ZINC',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 6,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 64,
#                 'dataset': 'ZINC',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': True,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'matrix_type': 'L',
#                 'n_heads': 8,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 8,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': True,
#                 'readout': 'mean',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/molecules_graph_regression_b64-lnorm-lap',
#  'params': {'batch_size': 64,
#             'epochs': 1000,
#             'init_lr': 0.001,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 15,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'seed_array': [41],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/molecules_graph_regression_GraphTransformer_ZINC_500k.json --layer_norm True --batch_norm False --edge_feat True --batch_size 64 --init_lr 0.001 --L 6 --hidden_dim 80 --out_dim 80 --matrix_type L --job_note b64-lnorm-lap --param_values 4 8 16 --rand_pos_enc True
