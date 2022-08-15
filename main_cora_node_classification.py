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
from pprint import pprint

from tensorboardX import SummaryWriter
from db import store_results
from utils.main_utils import DotDict, gpu_setup, view_model_param, get_logger, add_args, setup_dirs, get_parameters, get_net_params

logger = None


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.cora_node_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
from data.cora import CoraDataset
from train.train_cora_node_classification import train_epoch, evaluate_network



"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, config_file, config):
    avg_test_acc = []
    avg_train_acc = []
    avg_convergence_epochs = []

    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if net_params['pos_enc']:
        print('hi')
        logger.info("[!] Adding Laplacian graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])
    
    if net_params['adj_enc']:
        logger.info("[!] Adding adjacency matrix graph positional encoding.")
        dataset._add_adj_encodings(net_params['pos_enc_dim'])
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    train_masks = dataset.train_masks
    val_masks = dataset.val_masks
    test_masks = dataset.test_masks
    labels = dataset.labels.to(device)
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    try:
        for split_num in range(len(train_masks)):
            train_mask = train_masks[split_num].to(device)
            val_mask = val_masks[split_num].to(device)
            test_mask = test_masks[split_num].to(device)

            graph = dataset.graph.to(device)

            print("Total number of nodes: ", graph.number_of_nodes())
            print("Total number of edges: ", graph.number_of_edges())

            node_feat = graph.ndata['feat'].to(device)
            edge_feat = graph.edata['feat'].to(device)

            log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])

            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])
        
            logger.info(f"Split number: {split_num}", )
            logger.info(f"Training Nodes: { train_mask.int().sum().item()}",)
            logger.info(f"Validation Nodes: { val_mask.int().sum().item()}",)
            logger.info(f"Test Nodes: {test_mask.int().sum().item()}",)
            logger.info(f"Number of Classes: { net_params['n_classes']}",)

            model = gnn_model(MODEL_NAME, net_params)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=params['lr_reduce_factor'],
                                                            patience=params['lr_schedule_patience'],
                                                            verbose=True)
            
            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], [] 

            # At any point you can hit Ctrl + C to break out of training early.
            best_test_acc = -1.0
            best_train_acc = -1.0
            for epoch in range(params['epochs']):

                logger.info(f'Epoch {epoch + 1}/{params["epochs"]}')
                start = time.time()

                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, graph, node_feat, edge_feat, train_mask, labels, epoch)

                epoch_val_loss, epoch_val_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, val_mask, labels, epoch)
                _, epoch_test_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, test_mask, labels, epoch)        

                if epoch_test_acc > best_test_acc:
                    best_test_acc = epoch_test_acc
                    best_train_acc = epoch_train_acc
                #     model_dir = os.path.join(root_ckpt_dir, "MODELS_")
                #     if not os.path.exists(model_dir):
                #         os.makedirs(model_dir)
                #     fname = f"/best_model{best_test_acc:.4f}_{params['job_num']}.pt"
                #     torch.save(model.state_dict(), model_dir + fname)
                #     logger.info(f"Saving best model with test accuracy: {best_test_acc:.4f} to {model_dir}")

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

                t = time.time() - start
                lr = optimizer.param_groups[0]['lr']
                train_loss = epoch_train_loss
                val_loss = epoch_val_loss
                train_acc = epoch_train_acc
                val_acc = epoch_val_acc
                test_acc = epoch_test_acc

                logger.info(f"""\tTime: {t:.2f}s, LR: {lr:.5f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},
                            Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}""")

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
                    logger.info("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    logger.info('-' * 89)
                    logger.info("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
            
            
            _, test_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, test_mask, labels, epoch)   
            _, train_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, train_mask, labels, epoch)    

            avg_test_acc.append(test_acc)
            avg_train_acc.append(train_acc)
            avg_convergence_epochs.append(epoch)

            logger.info("Test Accuracy: {:.4f}".format(test_acc))
            logger.info("Best Test Accuracy: {:.4f}".format(best_test_acc))
            logger.info("Train Accuracy: {:.4f}".format(train_acc))
            logger.info("Best Train Accuracy Corresponding to Best Test Accuracy: {:.4f}".format(best_train_acc))
            logger.info("Convergence Time (Epochs): {:.4f}".format(epoch))
            writer.close()

    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early because of KeyboardInterrupt')

    logger.info("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    logger.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    params['config'] = config_file
    params['model'] = MODEL_NAME
    params['dataset'] = DATASET_NAME
    store_results(params, net_params, {
        'best_test_metric': best_test_acc,
        'best_train_metric': best_train_acc,
        'final_test_metric': test_acc,
        'final_train_metric': train_acc,
        'epochs_to_convergence': epoch,
        'total_time_taken': time.time()-t0,
        'avg_time_per_epoch': np.mean(per_epoch_time),
        'train_loss': epoch_train_losses,
        'lr': optimizer.param_groups[0]['lr']
    }, config)


    logger.info(f"TOTAL TIME TAKEN: {((time.time()-t0)/3600):.4f}hrs")
    logger.info(f"AVG TIME PER EPOCH: {np.mean(per_epoch_time):.4f}s")
    logger.info(f"AVG CONVERGENCE Time (Epochs): {np.mean(np.array(avg_convergence_epochs)):.4f}")

    # Final test accuracy value averaged over 20-split
    logger.info(f"""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {np.mean(np.array(avg_test_acc))*100:.4f} with s.d. {(np.std(avg_test_acc)*100):.4f}""")
    logger.info(f"\nAll splits Test Accuracies: {avg_test_acc}\n")
    logger.info(f"""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {np.mean(np.array(avg_train_acc))*100:.4f} with s.d. {(np.std(avg_train_acc)*100):.4f}""")
    logger.info(f"\nAll splits Train Accuracies: {avg_train_acc}\n")

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(test_acc))*100, np.mean(np.array(train_acc))*100, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))



def main(args):    
    """
        USER CONTROLS
    """

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
    # net_params = config['net_params']
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

    net_params['adj_enc'] = args.adj_enc
    net_params['dataset'] = DATASET_NAME
    net_params['matrix_type'] = args.matrix_type
    net_params['pow_of_mat'] = args.pow_of_mat

    # Superpixels
    net_params['in_dim'] = dataset.n_features
    # net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    net_params['n_classes'] = dataset.n_classes

    logger.info(net_params)
    logger.info(params)

    dirs = setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config)

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, gnn_model, logger)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, args.config, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    main(args) 
