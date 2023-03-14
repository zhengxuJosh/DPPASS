export cur_dir=`pwd`
export run_file="train.py"
export save_exp_name="xxxx"

python3 -m torch.distributed.launch --nproc_per_node=4 ${run_file} --save_root="${cur_dir}/exp/${save_exp_name}/" \
--lr=0.00006 --batch-size=2 --iterations=100000 \
\
--lamda xx \
2>&1 \
| tee "log/${save_exp_name}.`date`"
