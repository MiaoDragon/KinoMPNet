cd ..
python neural_train_waypoint_pos_vel.py --model_path /media/arclabdl1/HD1/YLmiao/results/KMPnet_res/ --no_env 10 --no_motion_paths 1800 --total_input_size 8 \
--AE_input_size 32 --mlp_input_size 136 --output_size 4 --learning_rate 0.001 --device 1 --num_epochs 10000 \
--batch_size 256 --path_folder ./data/cartpole_obs/ --path_file path.pkl --start_epoch 0 --env_type cartpole_obs_2 \
--world_size 30.0 40.0 3.141592653589793 2.0 --direction 0 --opt Adagrad --num_steps 200 --loss l1_smooth
cd exp
#10x800
