cd ..
python neural_test.py --model_path /media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_5_lr0.010000_SGD/ --seen_N 10 --seen_NP 10 --total_input_size 8 \
--AE_input_size 32 --mlp_input_size 136 --output_size 4 --learning_rate 0.01 --device 1 --start_epoch 5000 \
--data_folder ./data/acrobot_obs/ --env_type acrobot_obs_5 \
--world_size 3.141592653589793 3.141592653589793 6.0 6.0 --opt SGD
cd exp
#10x800
