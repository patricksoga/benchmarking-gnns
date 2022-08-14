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
from nets.COLLAB_edge_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset



"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    #assert net_params['self_loop'] == False, "No self-loop support for %s dataset" % DATASET_NAME
    
    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding Laplacian graph positional encoding",net_params['pos_enc_dim'])
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:',time.time()-t0)
        
    graph = dataset.graph
    
    evaluator = dataset.evaluator
    
    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg
        
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
    
    print("Graph: ", graph)
    print("Training Edges: ", len(train_edges))
    print("Validation Edges: ", len(val_edges) + len(val_edges_neg))
    print("Test Edges: ", len(test_edges) + len(test_edges_neg))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses = []
    epoch_train_hits, epoch_val_hits = [], [] 
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        raise NotImplementedError # gave OOM while preparing dense tensor
    else:
        # import train functions for all other GCNs
        from train.train_COLLAB_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        monet_pseudo = None
        if MODEL_NAME == "MoNet":
            print("\nPre-computing MoNet pseudo-edges")
            # for MoNet: computing the 'pseudo' named tensor which depends on node degrees
            us, vs = graph.edges()
            # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
            monet_pseudo = [ 
                [1/np.sqrt(graph.in_degree(us[i])+1), 1/np.sqrt(graph.in_degree(vs[i])+1)] 
                    for i in range(graph.number_of_edges())
            ]
            monet_pseudo = torch.Tensor(monet_pseudo)
        
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)    

                start = time.time()
                    
                epoch_train_loss, optimizer = train_epoch(model, optimizer, device, graph, train_edges, params['batch_size'], epoch, monet_pseudo)
                
                epoch_train_hits, epoch_val_hits, epoch_test_hits = evaluate_network(
                    model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_train_hits.append(epoch_train_hits)
                epoch_val_hits.append(epoch_val_hits)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                
                writer.add_scalar('train/_hits@10', epoch_train_hits[0]*100, epoch)
                writer.add_scalar('train/_hits@50', epoch_train_hits[1]*100, epoch)
                writer.add_scalar('train/_hits@100', epoch_train_hits[2]*100, epoch)
                
                writer.add_scalar('val/_hits@10', epoch_val_hits[0]*100, epoch)
                writer.add_scalar('val/_hits@50', epoch_val_hits[1]*100, epoch)
                writer.add_scalar('val/_hits@100', epoch_val_hits[2]*100, epoch)
                
                writer.add_scalar('test/_hits@10', epoch_test_hits[0]*100, epoch)
                writer.add_scalar('test/_hits@50', epoch_test_hits[1]*100, epoch)
                writer.add_scalar('test/_hits@100', epoch_test_hits[2]*100, epoch)
                
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)   

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, train_hits=epoch_train_hits[1], 
                              val_hits=epoch_val_hits[1], test_hits=epoch_test_hits[1]) 

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

                scheduler.step(epoch_val_hits[1])

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # # Stop training after params['max_time'] hours
                # if time.time()-t0 > params['max_time']*3600:
                #     print('-' * 89)
                #     print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                #     break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    
    train_hits, val_hits, test_hits = evaluate_network(
        model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo)

    print(f"Test:\nHits@10: {test_hits[0]*100:.4f}% \nHits@50: {test_hits[1]*100:.4f}% \nHits@100: {test_hits[2]*100:.4f}% \n")
    print(f"Train:\nHits@10: {train_hits[0]*100:.4f}% \nHits@50: {train_hits[1]*100:.4f}% \nHits@100: {train_hits[2]*100:.4f}% \n")
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST HITS@10: {:.4f}\nTEST HITS@50: {:.4f}\nTEST HITS@100: {:.4f}\nTRAIN HITS@10: {:.4f}\nTRAIN HITS@50: {:.4f}\nTRAIN HITS@100: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f}hrs\nAverage Time Per Epoch: {:.4f}s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_hits[0]*100, test_hits[1]*100, test_hits[2]*100, train_hits[0]*100, train_hits[1]*100, train_hits[2]*100,
                  epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
    




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

    global logger
    logger = get_logger(net_params['log_file'])


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
    if args.layer_type is not None:
        net_params['layer_type'] = layer_type
    net_params['dataset'] = DATASET_NAME

      
    
    # COLLAB
    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[-1]
    net_params['in_dim_edge'] = dataset.graph.edata['feat'].shape[-1]
    net_params['n_classes'] = 1  # binary prediction
    
    dirs = setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config)

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, gnn_model, logger)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

main()    
