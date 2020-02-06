cd ..
python neural_train_waypoint.py --model_path ./results/ --no_env 1 --no_motion_paths 100 --total_input_size 8 \
--AE_input_size 0 --mlp_input_size 8 --output_size 4 --learning_rate 0.01 --device 0 --num_epochs 1 \
--batch_size 10 --path_folder ./data/cartpole/ --start_epoch 0 --env_type cartpole \
--world_size 30.0 40.0 3.141592653589793 2.0 --obs_file ./data/cartpole_obs/obs.pkl \
--obc_file ./data/cartpole_obs/obc.pkl
cd exp
