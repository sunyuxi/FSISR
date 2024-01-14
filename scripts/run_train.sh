#hashbit_arr=(24 16 8 32 48 64)
hashbit_arr=(24)
lambda0=1
gpu=1
for hashbit in "${hashbit_arr[@]}"
do
  echo ${hashbit}
  echo ${lambda0}
  export CUDA_VISIBLE_DEVICES=${gpu}; python train.py --task_name 'ours' --hash_bit ${hashbit} --lambda0 ${lambda0} > logs/train_h${hashbit}_l0_${lambda0}_alexnet.log 2>&1 &
  wait
done
