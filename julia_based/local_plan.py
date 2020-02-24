import julia
from julia.api import Julia
from utils import load_data

if __name__ == '__main__':
    jl = Julia(compiled_modules=False)
    j = julia.Julia(runtime="julia")

    # model = 'cartpole'
    model = 'cartpole'
    cem = j.include("CEM_{}.jl".format(model))

    data = load_data(model, 1)
    path = data['path']
    i_node = 1
    for i_node in range(len(path)-1):# [i_node]:#
        start = path[i_node]
        goal = path[i_node+1]
        u, t = cem(start, goal)
        print(u, t)
