cd ..
cd ..
python sst_test_compare_with_mpnet.py \
--data_folder '/media/arclabdl1/HD1/YLmiao/data/kinodynamic/cartpole_obs/' --env_type cartpole_obs \
--opt SGD --num_steps 200 --seen_N 10 --seen_NP 200 --seen_s 0 --seen_sp 1800 --num_iter 5000000
cd exp
cd test