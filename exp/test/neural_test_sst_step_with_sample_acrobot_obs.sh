cd ..
cd ..
python neural_test_c++_sst_step_with_sample.py --learning_rate 0.01 \
--data_folder './data/acrobot_obs/' --start_epoch 3400 --env_type acrobot_obs --world_size 3.141592653589793 3.141592653589793 6.0 6.0 \
--opt SGD --num_steps 20 --seen_N 10 --seen_NP 100 --seen_s 0 --seen_sp 800 --cost_threshold 1.2
cd exp
cd test