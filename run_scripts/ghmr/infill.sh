cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung


pretrained_model=/home/zhiyin/iterative-editing-release/mdm/meos/uncond/model000280000.pt

for data in amasshml_FcShapeAxyzAvel; do 

expname=${cur_fname}-${db}-${data}
cmd="python -m gthmr.inpaint \
		--data_config_path gthmr/emp_train/config/data/${data}.yml \
		--save_dir gthmr/results/${expname} \
		--model_path $pretrained_model 
"


if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
# break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${cur_fname}-${partition}
#SBATCH --output=slurm_logs/${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/${cur_fname}-${partition}-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p ${partition}
#SBATCH --time=48:00:00

#necessary env
# source /home/users/wangkua1/setup_rl.sh
echo \"$cmd\"
eval \"$cmd\"
"

fi

done