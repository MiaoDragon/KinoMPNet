include("cartpole.jl")
using Random
import Statistics
function cem(start, goal, weights, μ_u, σ_u, μ_t, σ_t, verbose=false)
    rng = MersenneTwister(1234);
    n_t = 10
    n_sample = 100
    n_elite = 10

    loss_min = Inf
    best_u = ones(n_t)
    best_t = ones(n_t)
    
    count = 0
    @time begin
        for _ = 1:100
            # global μ_u, σ_u, μ_t, σ_t, loss_min, n_t, start, goal
            u = σ_u .* randn(rng, Float64, (n_sample, n_t)) .+ μ_u
            t = σ_t .* (randn(rng, Float64, (n_sample, n_t))) .+ μ_t
            t[t .< 0] .= 0
            loss = zeros(n_sample)
            x = deepcopy(start)
            for i = 1:n_sample
                x = deepcopy(start)
                for j = 1:n_t
                    x = propagate(x, u[i,j], convert(Int, round(t[i,j]/0.002)), 0.002)
                end
                loss[i] = sum(((x - goal).*weights).^2)
            end

            ind = sortperm(loss)[1:n_elite]
            μ_u = Statistics.mean(u[ind], dims=1)
            σ_u = Statistics.std(u[ind], dims=1)
            μ_t = Statistics.mean(t[ind], dims=1)
            σ_t = Statistics.std(t[ind], dims=1)
            
            temp_loss_min = min(loss...)

            if (loss_min - temp_loss_min)^2 < 1e-4
                count += 1
            end
            if temp_loss_min > 10
                μ_u = 0
                σ_u = 300
                μ_t = 0.1
                σ_t = 0.05
            end
            
            if temp_loss_min < loss_min
                loss_min = temp_loss_min
                best_u = u[ind[1],:]
                best_t = t[ind[1],:]
            end
            if verbose
                println("ep loss:", temp_loss_min,"\tmin loss:",loss_min)
            end

            if count > 5 || loss_min < 1e-1
                break
            end
            
            
        end
    end
    i = 1
    for i = 1:length(best_t)
        if best_t[i] > 0
            break
        end
    end

    return best_u[i], best_t[i], μ_u, σ_u, μ_t, σ_t
    
end
# end
# start = [-21.55734286,  -1.6512908 ,  -2.85039622,  -0.36047371]#[-21.45187569,   0.        ,  -2.82751449,   0.        ]
# goal = [-21.87288909,  -4.25320562,  -2.92477117,  -1.03945387]
# CEM.cem(start, goal)
