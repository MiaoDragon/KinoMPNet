cd ..
cd ..
python rrt_test.py \
--data_folder '/media/arclabdl1/HD1/YLmiao/data/kinodynamic/acrobot_obs_backup/' --env_type acrobot_obs \
--opt SGD --num_steps 20 --seen_N 10 --seen_NP 100 --seen_s 0 --seen_sp 900 --num_iter 3000000
cd exp
cd test