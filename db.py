import sqlite3

def init_db():
    con = sqlite3.connect('./data/experiments.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE experiments
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                config TEXT,
                model TEXT,
                dataset TEXT,
                out_dir TEXT,
                seed TEXT,
                epochs REAL,
                batch_size REAL,
                init_lr REAL,
                lr_reduce_factor REAL,
                lr_schedule_patience REAL,
                min_lr REAL,
                weight_decay REAL,
                print_epoch_interval REAL,
                L REAL,
                hidden_dim REAL,
                out_dim REAL,
                residual TEXT,
                edge_feat TEXT,
                readout TEXT,
                n_heads TEXT,
                in_feat_dropout TEXT,
                dropout TEXT,
                layer_norm TEXT,
                batch_norm TEXT,
                max_time REAL,
                num_train_data REAL,
                pos_enc_dim REAL,
                learned_pos_enc TEXT,
                rand_pos_enc TEXT,
                pos_enc TEXT,
                matrix_type TEXT,
                pow_of_mat REAL,
                adj_enc TEXT,
                num_initials REAL,
                pagerank TEXT,
                final_test_metric REAL,
                final_train_metric REAL,
                highest_test_metric REAL,
                highest_train_metric REAL,
                epochs_to_convergence REAL,
                total_time_taken REAL,
                avg_time_per_epoch REAL,
                final_lr REAL,
                num_parameters REAL
                )''')
    con.close()


def store_results(params, net_params, results_dict, config):
    highest_test_metric = results_dict['best_test_metric']
    highest_train_metric = results_dict['best_train_metric']
    final_test_metric = results_dict['final_test_metric']
    final_train_metric = results_dict['final_train_metric']
    epochs_to_convergence = results_dict['epochs_to_convergence']
    total_time_taken  = results_dict['total_time_taken']
    avg_time_per_epoch = results_dict['avg_time_per_epoch']
    lr = results_dict['lr']
    
    con = sqlite3.connect('./data/experiments.db')
    cur = con.cursor()

    cur.execute(f'''
        INSERT INTO experiments (config, model, dataset, out_dir, seed, epochs, batch_size, init_lr, lr_reduce_factor, lr_schedule_patience, min_lr, weight_decay, print_epoch_interval, L, hidden_dim, out_dim, residual, edge_feat, readout, n_heads, in_feat_dropout, dropout, layer_norm, batch_norm, max_time, num_train_data, pos_enc_dim, learned_pos_enc, rand_pos_enc, pos_enc, matrix_type, pow_of_mat, adj_enc, num_initials, pagerank, final_test_metric, final_train_metric, highest_test_metric, highest_train_metric, epochs_to_convergence, total_time_taken, avg_time_per_epoch, final_lr, num_parameters) values ({' '.join(['?,']*43) + '?'})''',
        (params['config'], params['model'], params['dataset'], config['out_dir'], params['seed'], params['epochs'], params.get('batch_size', -1), params['init_lr'], params['lr_reduce_factor'], params['lr_schedule_patience'], params['min_lr'], params['weight_decay'], params['print_epoch_interval'], net_params['L'], net_params['hidden_dim'], net_params['out_dim'], net_params['residual'], net_params.get('edge_feat', 'False'), net_params['readout'], net_params.get('n_heads', -1), net_params['in_feat_dropout'], net_params['dropout'], net_params.get('layer_norm', "N/A"), net_params['batch_norm'], params['max_time'], net_params.get('num_train_data', -1), net_params.get('pos_enc_dim', -1), net_params.get('learned_pos_enc', 'False'), net_params.get('rand_pos_enc', "False"), net_params.get('pos_enc', 'False'), net_params.get('matrix_type', 'N/A'), net_params.get('pow_of_mat', -1), net_params.get('adj_enc', 'False'), net_params.get('num_initials', -1), net_params.get('pagerank', 'False'), final_test_metric, final_train_metric, highest_test_metric, highest_train_metric, epochs_to_convergence, total_time_taken, avg_time_per_epoch, lr, int(net_params['total_param']))
    )

    con.commit()