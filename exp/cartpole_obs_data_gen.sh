cd ..
python data_generation.py --env_name cartpole_obs --N 2 --NP 2 \
--max_iter 100000 --path_folder ./data/cartpole_obs/ --path_file path.pkl \
--obs_file ./data/cartpole_obs/obs.pkl --obc_file ./data/cartpole_obs/obc.pkl
cd exp
# 100 x 5000