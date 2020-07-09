python py_model_to_cpp_waypoint.py --AE_input_size 32 --mlp_input_size 136 --output_size 4 --learning_rate 0.001 --device 0 \
--path_folder ../data/cartpole_obs/ --path_file path.pkl --start_epoch 2650 --env_type cartpole_obs --total_input_size 8 \
--world_size 30.0 40.0 3.141592653589793 2.0 --direction 0 --opt Adagrad --num_steps 200 --loss l1_smooth --multigoal 1
