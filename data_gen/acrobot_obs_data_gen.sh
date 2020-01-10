cd ..
python data_generation.py --env_name acrobot_obs --N 2 --NP 5 \
--max_iter 300000 --path_folder ../data/acrobot_obs/ \
--obs_file ../data/acrobot_obs/obs.pkl --obc_file ../data/acrobot_obs/obc.pkl
cd exp
# 100 x 2000
