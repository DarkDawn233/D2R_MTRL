algorithm=$1
env=$2
fix_or_random=$3

config_path="config/$env/$algorithm"_"$fix_or_random.json"
id_name="$env"_"$fix_or_random"
seed=$4

cuda=$5

cmd="python train.py --config $config_path --id $id_name --seed $seed"

echo "CUDA_VISIBLE_DEVICES=$cuda"
echo $cmd

CUDA_VISIBLE_DEVICES=$cuda $cmd

