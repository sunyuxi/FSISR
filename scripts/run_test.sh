#hashbit_arr=(24 16 8 32 48 64)
hashbit_arr=(24)
gpu=0
for hashbit in "${hashbit_arr[@]}"
do
  echo ${hashbit}
  export CUDA_VISIBLE_DEVICES=${gpu}; python AllDeepbaselines_test.py --task_name 'ours' --hash_bit ${hashbit} > logs/test_h${hashbit}_alexnet.log 2>&1 &
  wait
done
