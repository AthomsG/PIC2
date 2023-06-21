mkdir ./train_log/$1
export TF_CPP_MIN_LOG_LEVEL=3
CUDA_VISIBLE_DEVICES=0 python -u train.py --game_name $1 --update_target_every 2500 --reg tsallis --lambd 1.0  > ./train_log/$1/rqn_tsallis_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -u train.py --game_name $1 --update_target_every 2500 --reg tsallis --lambd 0.1  > ./train_log/$1/rqn_tsallis_0.1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -u train.py --game_name $1 --update_target_every 2500 --reg tsallis --lambd 0.01 > ./train_log/$1/rqn_tsallis_0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --game_name $1 --update_target_every 2500 --reg cosx --lambd 1.0  > ./train_log/$1/rqn_cosx_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --game_name $1 --update_target_every 2500 --reg cosx --lambd 0.1  > ./train_log/$1/rqn_cosx_0.1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --game_name $1 --update_target_every 2500 --reg cosx --lambd 0.01 > ./train_log/$1/rqn_cosx_0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --game_name $1 --update_target_every 2500 --reg expx --lambd 1.0  > ./train_log/$1/rqn_expx_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --game_name $1 --update_target_every 2500 --reg expx --lambd 0.1  > ./train_log/$1/rqn_expx_0.1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --game_name $1 --update_target_every 2500 --reg expx --lambd 0.01 > ./train_log/$1/rqn_expx_0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --game_name $1 --update_target_every 2500 --reg shannon --lambd 1.0  > ./train_log/$1/rqn_logx_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --game_name $1 --update_target_every 2500 --reg shannon --lambd 0.1  > ./train_log/$1/rqn_logx_0.1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --game_name $1 --update_target_every 2500 --reg shannon --lambd 0.01 > ./train_log/$1/rqn_logx_0.01.log 2>&1 &
