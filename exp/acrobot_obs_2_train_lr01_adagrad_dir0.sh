cd ..
python neural_train.py --model_path /media/arclabdl1/HD1/YLmiao/results/KMPnet_res/ --no_env 10 --no_motion_paths 800 --total_input_size 8 \
--AE_input_size 32 --mlp_input_size 136 --output_size 4 --learning_rate 0.01 --device 0 --num_epochs 500 \
--batch_size 32 --path_folder ./data/acrobot_obs/ --path_file path.pkl --start_epoch 0 --env_type acrobot_obs_2 \
--world_size 3.141592653589793 3.141592653589793 6.0 6.0 --direction 0 --opt Adagrad
cd exp
#10x800
