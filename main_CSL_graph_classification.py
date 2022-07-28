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

"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.CSL_graph_classification.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset


"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []
    avg_epochs = []

    t0 = time.time()
    per_epoch_time = []

    dataset = LoadData(DATASET_NAME)
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    if net_params['pos_enc']:
        print("[!] Adding Laplacian graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for split_number in range(5):
            
            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])

            model = gnn_model(MODEL_NAME, net_params)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], [] 

            # batching exception for Diffpool
            drop_last = True if MODEL_NAME == 'DiffPool' else False
            # drop_last = False

            
            if MODEL_NAME in ['RingGNN', '3WLGNN']:
                # import train functions specific for WL-GNNs
                from train.train_CSL_graph_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network
                from functools import partial # util function to pass pos_enc flag to collate function

                train_loader = DataLoader(trainset, shuffle=True, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
                val_loader = DataLoader(valset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
                test_loader = DataLoader(testset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))

            else:
                # import train functions for all other GCNs
                from train.train_CSL_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

                train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
                val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
                test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

        
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:

                    t.set_description('Epoch %d' % epoch)    

                    start = time.time()
                    
                    if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                    else:   # for all other models common train function
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    
                    #epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)

                    _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)
                    
                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)

                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                    
                    epoch_train_acc = 100.* epoch_train_acc
                    epoch_test_acc = 100.* epoch_test_acc
                    
                    t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)  

                    per_epoch_time.append(time.time()-start)

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
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
                    if time.time()-t0_split > params['max_time']*3600/10:       # Dividing max_time by 10, since there are 10 runs in TUs
                        print('-' * 89)
                        print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                        break

            _, test_acc = evaluate_network(model, device, test_loader, epoch)   
            _, train_acc = evaluate_network(model, device, train_loader, epoch)    
            avg_test_acc.append(test_acc)   
            avg_train_acc.append(train_acc)
            avg_epochs.append(epoch)

            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        
    
    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.3f}\n with test acc s.d. {:.3f}\nTRAIN ACCURACY averaged: {:.3f}\n with train s.d. {:.3f}\n\n
    Convergence Time (Epochs): {:.3f}\nTotal Time Taken: {:.3f} hrs\nAverage Time Per Epoch: {:.3f} s\n\n\nAll Splits Test Accuracies: {}\n\nAll Splits Train Accuracies: {}"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
                  np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100, np.mean(np.array(avg_epochs)),
               (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc, avg_train_acc))
        




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
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
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

    # CSL
    net_params['num_node_type'] = dataset.all.num_node_type
    net_params['num_edge_type'] = dataset.all.num_edge_type
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes
    
    # RingGNN
    if MODEL_NAME == 'RingGNN':
        num_nodes_train = [dataset.train[0][i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[0][i][0].number_of_nodes() for i in range(len(dataset.test))]
        num_nodes = num_nodes_train + num_nodes_test
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
        
    # RingGNN, 3WLGNN
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        if net_params['pos_enc']:
            net_params['in_dim'] = net_params['pos_enc_dim']
        else:
            net_params['in_dim'] = 1

    dirs = setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config)

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, gnn_model, logger)
    train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)

main()    
