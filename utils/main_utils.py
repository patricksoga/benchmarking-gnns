import os
import torch
import numpy as np
import logging
import time
import sqlite3

def get_logger(logfile=None):
    _logfile = logfile if logfile else './DEBUG.log'
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s(): %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id, logger):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        logger.info(f'cuda available with GPU: {torch.cuda.get_device_name(0)}')
        device = torch.device("cuda")
    else:
        logger.info('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params, gnn_model, logger):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    logger.info("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    logger.info(f'MODEL/Total parameters: {MODEL_NAME}, {total_param}')
    return total_param


def add_args(parser):
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--num_train_data', help="Please give a value for num_train_data")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--job_num', help="Please give a value for job number")
    parser.add_argument('--learned_pos_enc', help="Please give a value for learned_pos_enc")
    parser.add_argument('--rand_pos_enc', help="Whether to use a random automata PE")
    parser.add_argument('--pos_enc', help="Whether to use Laplacian PE or not")
    parser.add_argument('--matrix_type', type=str, help="Type of matrix to use in automata PE")
    parser.add_argument('--pow_of_mat', type=int, default=1, help="Highest power of adjacency matrix to use in automata PE")
    parser.add_argument('--log_file', type=str, default="./DEBUG.log")
    parser.add_argument('--adj_enc', action='store_true', help="Use adjacency matrix eigenvectors for PE")
    parser.add_argument('--num_initials', help="Number of initial weight vectors for automata PE")
    parser.add_argument('--full_graph', help="Use full graph for graph transformer")
    parser.add_argument('--power_method', help="Use power method for graph transformer automata PE")
    parser.add_argument('--power_iters', help="Number of power method iterations for graph transformer automata PE")
    parser.add_argument('--seed_array', help="Array of seeds to use for testing", nargs='+', type=int)
    parser.add_argument('--save_name', help="Name of saved results file")
    parser.add_argument('--rw_pos_enc', help="Use random walk PE for graph transformer")
    parser.add_argument('--partial_rw_pos_enc', help="Use partial random walk PE for graph transformer")
    parser.add_argument('--diag', help="Use diagonal matrix for automaton PE")
    parser.add_argument('--cat_gape', help="Use concatenation for graph transformer (GAPE)")
    parser.add_argument('--n_gape', help="Use multiple GAPE encodings for graph transformer")
    parser.add_argument('--gape_pooling', help="Type of pooling for GAPE for graph transformer (GAPE)")

    parser.add_argument('--spectral_attn', help="Use spectral attention for graph transformer")
    parser.add_argument('--lpe_layers', help="Number of layers for graph transformer (spectral attention)")
    parser.add_argument('--lpe_dim', help="Dimension of graph transformer PE (spectral attention)")
    parser.add_argument('--lpe_n_heads', help="Number of heads for spectral attention PE (spectral attention)")

    parser.add_argument('--in_deg_centrality', help="Max in-degree centrality PE for graph transformer (Graphormer)")
    parser.add_argument('--out_deg_centrality', help="Max out-degree centrality PE for graph transformer (Graphormer)")
    parser.add_argument('--spd_len', help="Max shortest path distance for use as spatial PE for graph transformer (Graphormer)")

    parser.add_argument('--pagerank')
    return parser

def setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config):
    dir_str = ""
    if args.job_num:
        dir_str = args.job_num + "_"

    if args.pos_enc_dim is not None:
        dir_str += args.pos_enc_dim + "_"

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + dir_str + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + dir_str + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + dir_str + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + dir_str + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    return dirs


def get_parameters(config, args):
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.job_num is not None:
        params['job_num'] = int(args.job_num)
    if args.seed_array is not None:
        params['seed_array'] = args.seed_array
    else:
        params['seed_array'] = [41]
    if args.save_name is not None:
        params['save_name'] = args.save_name

    return params


def get_net_params(config, args, device, params, DATASET_NAME):
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params.get('batch_size', -1)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)

    if args.learned_pos_enc is not None:
        # net_params['learned_pos_enc'] = args.learned_pos_enc
        net_params['learned_pos_enc'] = True if args.learned_pos_enc=='True' else False
    if args.rand_pos_enc is not None:
        net_params['rand_pos_enc'] = True if args.rand_pos_enc=='True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc =='True' else False
    if args.rw_pos_enc is not None:
        net_params['rw_pos_enc'] = True if args.rw_pos_enc=='True' else False
    elif 'rw_pos_enc' not in config:
        net_params['rw_pos_enc'] = False

    if args.partial_rw_pos_enc is not None and 'partial_rw_pos_enc' not in net_params:
        net_params['partial_rw_pos_enc'] = True if args.partial_rw_pos_enc=='True' else False

    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.num_initials is not None:
        net_params['num_initials'] = int(args.num_initials)
    if args.pagerank is not None:
        net_params['pagerank'] = True if args.pagerank=='True' else False
    if args.full_graph is not None:
        net_params['full_graph'] = True if args.full_graph=='True' else False
    if args.power_method is not None:
        net_params['power_method'] = True if args.power_method=='True' else False
    elif 'power_method' not in config:
        net_params['power_method'] = False
    if args.power_iters is not None:
        net_params['power_iters'] = int(args.power_iters)
    if args.diag is not None:
        net_params['diag'] = True if args.diag=='True' else False
    elif 'diag' not in config and 'diag' not in net_params:
        net_params['diag'] = False
    
    if args.pow_of_mat is not None and 'pow_of_mat' not in net_params:
        net_params['pow_of_mat'] = int(args.pow_of_mat)
    elif args.pow_of_mat in config:
        net_params['pow_of_mat'] = int(config['pow_of_mat'])

    net_params['adj_enc'] = args.adj_enc
    net_params['dataset'] = DATASET_NAME
    if args.matrix_type is not None:
        net_params['matrix_type'] = args.matrix_type
    elif 'matrix_type' not in config and 'matrix_type' not in net_params:
        net_params['matrix_type'] = 'A'
    elif 'matrix_type' in net_params:
        net_params['matrix_type'] = net_params['matrix_type']
    else:
        net_params['matrix_type'] = config['matrix_type']
    
    if args.spectral_attn is not None:
        net_params['spectral_attn'] = True if args.spectral_attn=='True' else False
    elif 'spectral_attn' not in config and 'spectral_attn' not in net_params:
        net_params['spectral_attn'] = False
    elif 'spectral_attn' in net_params:
        net_params['spectral_attn'] = net_params['spectral_attn']
    else:
        net_params['spectral_attn'] = config['spectral_attn']

    if args.cat is not None:
        net_params['cat_gape'] = True if args.cat_gape=='True' else False
    elif 'cat_gape' not in config and 'cat_gape' not in net_params:
        net_params['cat_gape'] = False
    elif 'cat_gape' in net_params:
        net_params['cat_gape'] = net_params['cat_gape']
    else:
        net_params['cat_gape'] = config['cat_gape']

    if args.lpe_layers is not None:
        net_params['lpe_layers'] = int(args.lpe_layers)
    if args.lpe_n_heads is not None:
        net_params['lpe_n_heads'] = int(args.lpe_n_heads)

    if args.n_gape is not None:
        net_params['n_gape'] = int(args.n_gape)

    if args.spd_len is not None:
        net_params['spd_len'] = int(args.spd_len)

    if args.in_deg_centrality is not None:
        net_params['in_deg_centrality'] = int(args.in_deg_centrality)

    if args.out_deg_centrality is not None:
        net_params['out_deg_centrality'] = int(args.out_deg_centrality)
    
    if args.gape_pooling is not None:
        net_params['gape_pooling'] = args.gape_pooling

    # net_params['pow_of_mat'] = args.pow_of_mat

    return net_params