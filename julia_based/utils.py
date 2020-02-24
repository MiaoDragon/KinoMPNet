import pickle

def load_data(model, id):
    filepath = lambda var: "../trajs/{model}/{var}_{id}.pkl".format(model=model, var=var, id=id)
    load_pkl = lambda var: pickle.load(open(filepath(var), "rb"))
    keys = ["control", "path", "start_goal", "time", 'cost']
    return dict(zip(keys, [load_pkl(key) for key in keys]))

if __name__ == "__main__":
    dc = load_data('cartpole', 1)
    da = load_data('acrobot', 1)