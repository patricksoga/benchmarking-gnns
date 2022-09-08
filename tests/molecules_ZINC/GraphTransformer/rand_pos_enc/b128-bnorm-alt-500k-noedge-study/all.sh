# pos_enc_dim=(2 4 8 16 32 64 128 256)
pos_enc_dim=(8)
curr_dir=$(pwd)

for i in ${pos_enc_dim[@]}; 
do 
    cd ~/Documents/projects/benchmarking-gnns;
    fname=${curr_dir}/b128-bnorm-alt-noedge-fg-study_${i}_${pos_enc_dim[${i}]}_DEBUG.log;
    if [ $i -eq 256 ] 
    then
        python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-fg-study.json --job_num ${i} --pos_enc_dim ${i} --log_file $fname --full_graph False --L 9
    else
        python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-500k-noedge-study.json --job_num ${i} --pos_enc_dim ${i} --log_file $fname --full_graph False
    fi
done