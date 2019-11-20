cd ..
python neural_sample.py --model_path ./results/ --total_input_size 4 \
--AE_input_size 0 --mlp_input_size 4 --output_size 2 --learning_rate 0.01 --device 0 \
--seen_N 1 --seen_NP 10 --seen_s 0 --seen_sp 0 --unseen_N 0 --unseen_NP 0 --unseen_s 0 --unseen_sp 0 \
--path_folder ./data/pendulum/ --start_epoch 20 --env_type pendulum \
--world_size 3.141592653589793 7.0 --obs_file ./data/cartpole_obs/obs.pkl \
--obc_file ./data/cartpole_obs/obc.pkl
cd exp
