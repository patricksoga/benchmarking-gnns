




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

from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.main_utils import DotDict, gpu_setup, view_model_param, get_logger, add_args, setup_dirs, get_parameters, get_net_params
logger = None

"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.WikiCS_node_classification.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset
from train.train_WikiCS_node_classification import train_epoch, evaluate_network # import train functions




"""
    GPU Setup
"""












"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    
    avg_test_acc = []
    avg_train_acc = []
    avg_convergence_epochs = []
    
    t0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = dataset.name
    
    if MODEL_NAME in ['MoNet']:
        if net_params['pos_enc']:
            print("[!] Adding Laplacian graph positional encoding.")
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:',time.time()-t0)
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    train_masks = dataset.train_masks
    val_masks = dataset.val_masks
    test_mask = dataset.test_mask.to(device)
    labels = dataset.labels.to(device)
        
    
    
    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
   
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for split_num in range(len(train_masks)):
            t0_split = time.time()
            
            train_mask = train_masks[split_num].to(device)
            val_mask = val_masks[split_num].to(device)
            graph = dataset.g.to(device)
            print("Total num nodes: ", graph.number_of_nodes())
            print("Total num edges: ", graph.number_of_edges())
            node_feat = graph.ndata['feat'].to(device)
            edge_feat = graph.edata['feat'].long().to(device)
            
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_num))
            writer = SummaryWriter(log_dir=log_dir)
            
            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            print("Split number: ", split_num)
            print("Training Nodes: ", train_mask.int().sum().item())
            print("Validation Nodes: ", val_mask.int().sum().item())
            print("Test Nodes: ", test_mask.int().sum().item())
            print("Number of Classes: ", net_params['n_classes'])

            model = gnn_model(MODEL_NAME, net_params)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs, epoch_test_accs = [], [], []
    
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:

                    t.set_description('Epoch %d' % epoch)

                    start = time.time()

                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, graph, node_feat, edge_feat, train_mask, labels, epoch)

                    epoch_val_loss, epoch_val_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, val_mask, labels, epoch)
                    _, epoch_test_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, test_mask, labels, epoch)        

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)
                    epoch_test_accs.append(epoch_test_acc)

                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)

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
                        print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                        break

                    # Stop training after params['max_time'] hours
                    if time.time()-t0_split > params['max_time']*3600/20: # Dividing max_time by 20, since there are 20 splits in WikiCS
                        print('-' * 89)
                        print("Max_time for training one split elapsed {:.2f} hours, so stopping".format(params['max_time']))
                        break

            _, test_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, test_mask, labels, epoch)   
            _, train_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, train_mask, labels, epoch)    
            
            avg_test_acc.append(test_acc)   
            avg_train_acc.append(train_acc)
            avg_convergence_epochs.append(epoch)
            
            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
            print("Convergence Time (Epochs): {:.4f}".format(epoch))
            
            writer.close()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    

    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(avg_convergence_epochs))))
    
    # Final test accuracy value averaged over 20-split
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)


    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
    Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
                  np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100,
                  np.mean(avg_convergence_epochs), np.std(avg_convergence_epochs),
               (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc))

        




def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
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
    #net_params['batch_size'] = params['batch_size']
    net_params = get_net_params(config, args, device, params, DATASET_NAME)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
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

    # SBM
    net_params['in_dim'] = dataset.n_feats
    net_params['n_classes'] = dataset.num_classes
    
    dirs = setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config)



    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, gnn_model, logger)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

main()    
