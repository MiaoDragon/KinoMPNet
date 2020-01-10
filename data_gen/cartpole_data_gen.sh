cd ..
python data_generation.py --env_name cartpole --N 1 --NP 5000 \
--max_iter 200000 --path_folder ./data/cartpole/ \
--obs_file ./data/cartpole_obs/obs.pkl --obc_file ./data/cartpole_obs/obc.pkl
cd exp
# 100 x 2000
