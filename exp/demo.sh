cd ..
python neural_sampler.py --model_path ./results/ --no_env 1 --no_motion_paths 100 --total_input_size 8 \
--AE_input_size 0 --mlp_input_size 8 --output_size 4 --learning_rate 0.01 --device 0 --num_epochs 1 \
--batch_size 10 --data_path ./data/cartpole/ --start_epoch 0