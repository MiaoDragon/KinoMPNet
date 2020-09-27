cd ..
cd ..
python neural_test_c++_sst_step_with_sample.py --learning_rate 0.001 \
--data_folder './data/cartpole_obs/' --start_epoch 2650 --env_type cartpole_obs --world_size 30.0 40.0 3.141592653589793 2.0 \
--opt SGD --num_steps 200 --seen_N 10 --seen_NP 100 --seen_s 0 --seen_sp 800 --cost_threshold 1.2
cd exp
cd test