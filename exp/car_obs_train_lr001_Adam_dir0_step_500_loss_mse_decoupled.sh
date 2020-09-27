cd ..
python neural_train_waypoint.py --model_path /media/arclabdl1/HD1/YLmiao/results/KMPnet_res/ --no_env 10 --no_motion_paths 800 --total_input_size 6 \
--AE_input_size 32 --mlp_input_size 38 --output_size 3 --learning_rate 0.001 --device 1 --num_epochs 10000 \
--batch_size 256 --path_folder ./data/car_obs/ --path_file path.pkl --start_epoch 0 --env_type car_obs \
--world_size 25.0 35.0 3.141592653589793 --direction 0 --opt Adam --num_steps 500 --loss mse_decoupled
cd exp
#10x800
