cd ..
cd ..
python sst_test.py \
--data_folder '/media/arclabdl1/HD1/YLmiao/data/kinodynamic/car_obs/' --env_type car_obs \
--opt SGD --num_steps 500 --seen_N 10 --seen_NP 200 --seen_s 0 --seen_sp 800 --num_iter 500000
cd exp
cd test