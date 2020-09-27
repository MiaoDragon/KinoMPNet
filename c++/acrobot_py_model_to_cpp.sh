python py_model_to_cpp_waypoint.py --model_path /media/arclabdl1/HD1/YLmiao/results/KMPnet_res/ --no_env 1 --no_motion_paths 1 --total_input_size 8 \
--AE_input_size 32 --mlp_input_size 40 --output_size 4 --learning_rate 0.01 --device 0 \
--path_folder ../data/acrobot_obs/ --path_file path.pkl --start_epoch 2850 --env_type acrobot_obs \
--world_size 3.141592653589793 3.141592653589793 6.0 6.0 --direction 0 --opt SGD \
--num_steps 20 --loss mse --multigoal 0
