#!/bin/bash
#$ -N GraphTransformer_SBM_PATTERN_b26-lnorm-alt
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-5:1

pos_enc_dim=(0 2 4 8 32 64)
for SGE_TASK_ID in "${pos_enc_dim[@]}"; do
    fname=$(pwd)/b26-lnorm-alt_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
    touch $fname
done

for SGE_TASK_ID in "${pos_enc_dim[@]}"; do
    python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_PATTERN_b26-lnorm-alt.json --job_num ${SGE_TASK_ID} --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]} --log_file $fname &
done

# {'dataset': 'SBM_PATTERN',
#  'gpu': {'id': 0, 'use': True},
#  'model': 'GraphTransformer',
#  'net_params': {'L': 10,
#                 'adj_enc': False,
#                 'batch_norm': False,
#                 'batch_size': 26,
#                 'dataset': 'SBM_PATTERN',
#                 'dropout': 0.0,
#                 'full_graph': False,
#                 'gpu_id': 0,
#                 'hidden_dim': 80,
#                 'in_feat_dropout': 0.0,
#                 'layer_norm': True,
#                 'matrix_type': 'A',
#                 'n_heads': 8,
#                 'out_dim': 80,
#                 'pos_enc': False,
#                 'pow_of_mat': 1,
#                 'rand_pos_enc': True,
#                 'readout': 'mean',
#                 'residual': True,
#                 'self_loop': False,
#                 'wl_pos_enc': False},
#  'out_dir': 'out/SBMs_node_classification_b26-lnorm-alt',
#  'params': {'batch_size': 26,
#             'epochs': 1000,
#             'init_lr': 0.0005,
#             'lr_reduce_factor': 0.5,
#             'lr_schedule_patience': 10,
#             'max_time': 24,
#             'min_lr': 1e-06,
#             'print_epoch_interval': 5,
#             'seed': 41,
#             'weight_decay': 0.0}}



# Generated with command:
#python3 configure_tests.py --config ../configs/SBMs_node_clustering_GraphTransformer_PATTERN_500k.json --job_note b26-lnorm-alt --param_values 2 4 8 32 64 --batch_size 26 --rand_pos_enc True --batch_norm False --layer_norm True
