"""
    IMPORTING LIBS
"""
import numpy as np
import os
import time
import random
import glob
import argparse, json

import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.main_utils import DotDict, gpu_setup, view_model_param, get_logger, add_args, setup_dirs, get_parameters, get_net_params

logger = None


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.GraphTheoryProp_multitask.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset


"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if net_params['pos_enc']:
        print("[!] Adding Laplacian graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_mses, epoch_val_mses = [], [] 
    

    # import train functions for all other GCNs
    from train.train_GraphTheoryProp_multitask import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(params['epochs']):

            logger.info(f'Epoch {epoch + 1}/{params["epochs"]}')

            start = time.time()

            epoch_train_loss, epoch_train_log_mse, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

            epoch_val_loss, epoch_val_log_mse, __ = evaluate_network(model, device, val_loader, epoch)
            _, epoch_test_log_mse, __ = evaluate_network(model, device, test_loader, epoch)                
            
            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_mses.append(epoch_train_log_mse)
            epoch_val_mses.append(epoch_val_log_mse)

            writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            writer.add_scalar('train/_log_mse', epoch_train_log_mse, epoch)
            writer.add_scalar('val/_log_mse', epoch_val_log_mse, epoch)
            writer.add_scalar('test/_log_mse', epoch_test_log_mse, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            logger.info(f"""\tTime: {time.time()-start}s, LR: {optimizer.param_groups[0]['lr']:.4f}, 
                        train_loss: {epoch_train_loss:.4f}, train_log_mse: {epoch_train_log_mse:.4f},
                        val_loss: {epoch_val_loss:.4f},val_log_mse: {epoch_val_log_mse:.4f},
                        test_log_mse: {epoch_test_log_mse:.4f}""")
            
            per_epoch_time.append(time.time()-start)

            # Saving checkpoint
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

            files = glob.glob(ckpt_dir + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < epoch-1:
                    os.remove(file)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break
                
            # Stop training after params['max_time'] hours
            if time.time()-t0 > params['max_time']*3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_log_mse, specific_test_log_mse = evaluate_network(model, device, test_loader, epoch)
    _, train_log_mse, specific_train_log_mse = evaluate_network(model, device, train_loader, epoch)
    print("Test Log MSE: {:.4f}".format(test_log_mse))
    print("Specific Test Log MSE: {}".format(specific_test_log_mse))
    print("Train Log MSE: {:.4f}".format(train_log_mse))
    print("Specific Train Log MSE: {}".format(specific_train_log_mse))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST Log MSE: {:.4f}\nSpecific TEST Log MSE: {}\nTRAIN Log MSE: {:.4f}\nSpecific TRAIN Log MSE: {}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_log_mse, specific_test_log_mse, train_log_mse, specific_train_log_mse, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
               




def main():    
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    net_params = config['net_params']
    net_params['log_file'] = args.log_file

    global logger
    logger = get_logger(net_params['log_file'])

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'], logger)
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = get_parameters(config, args)
    # network parameters
    net_params = get_net_params(config, args, device, params, DATASET_NAME)

    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.num_train_data is not None:
        net_params['num_train_data'] = int(args.num_train_data)

    D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                   dataset.train.graph_lists])
    net_params['avg_d'] = dict(lin=torch.mean(D),
                               exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                               log=torch.mean(torch.log(D + 1)))
        
    dirs = setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config)

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, gnn_model, logger)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

main()    
