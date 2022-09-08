indices=(0 1 2 3 4 5 6 7)
pos_enc_dim=(2 4 8 16 32 64 128 256)
curr_dir=$(pwd)

for i in ${indices[@]}; 
do 
    cd ~/Documents/projects/benchmarking-gnns;
    fname=${curr_dir}/b32-bnorm-alt-study-lambda_${i}_${pos_enc_dim[${i}]}_DEBUG.log;
    if [ $i -eq 256 ] 
    then
        python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-study.json --job_num ${i} --pos_enc_dim ${pos_enc_dim[${i}]} --log_file $fname --full_graph False --L 9 --gpu_id ${indices[${i}]}
    else
        python3 main_SBMs_node_classification.py --config tests/test-configs/GraphTransformer_SBMs_SBM_CLUSTER_b32-bnorm-alt-study.json --job_num ${i} --pos_enc_dim ${pos_enc_dim[${i}]} --log_file $fname --full_graph False --gpu_id ${indices[${i}]}
    fi
done