#!/bin/bash
#$ -N GraphTransformer_CYCLES_b25-bnorm-alt-lap
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

pos_enc_dim=(0 40 64 128 180 256)
fname=$(pwd)/b25-bnorm-alt-lap_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-lap.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname


# {'dataset': 'CYCLES',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': True,
#                 'batch_size': 25,
#                 'dataset': 'CYCLES',
#                 'diag': False,
#                 'dropout': 0.0,
#                 'edge_feat': False,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': False,
#                 'learned_pos_enc': False,
#                 'matrix_type': 'L',
#                 'n_heads': 8,
#                 'num_train_data': 200,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pos_enc_dim': 20,
#                 'pow_of_mat': 1,
#                 'power_method': False,
#                 'rand_pos_enc': True,
#                 'readout': 'sum',
#                 'residual': True,
#                 'rw_pos_enc': False,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/CYCLES_graph_classification_b25-bnorm-alt-lap',
#  'params': {'batch_size': 25,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'seed_array': [41],
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/CYCLES_graph_classification_GraphTransformer_CYCLES_500k.json --batch_size 25 --job_note b25-bnorm-alt-lap --rand_pos_enc True --param_values 40 64 128 180 256 --batch_norm True --layer_norm False --edge_feat False --matrix_type L
