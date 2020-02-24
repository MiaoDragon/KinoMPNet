include("quadrotor.jl")
using Random
import Statistics
function cem(start, goal, weights, μ_u, σ_u, μ_t, σ_t, n_t=10, verbose=false)
    rng = MersenneTwister(1234);
    n_sample = 100
    n_elite = 10

    loss_min = Inf
    best_u = ones(n_t, 4)
    best_t = ones(n_t)  
    
    count = 0
    @time begin
        for _ = 1:100
            u = σ_u .* randn(rng, Float64, (n_sample, n_t, 4)) .+ μ_u
            u0s = u[:,:,1]
            u0s[u0s .< 5 ] .= 5
            u0s[u0s .> 15 ] .= 15
            u[:,:,1] = u0s
            for i=2:4
                ui = u[:,:,i]
                ui[ui .> 1] .= 1
                ui[ui .< -1] .= -1
                u[:,:,i] = ui
            end
            t = σ_t .* (randn(rng, Float64, (n_sample, n_t))) .+ μ_t
            t[t .< 0] .= 0
            loss = zeros(n_sample)
            x = deepcopy(start)
            for i = 1:n_sample
                x = deepcopy(start)
                for j = 1:n_t
                    x = propagate(x, u[i,j,:], convert(Int, round(t[i,j]/0.02)), 0.02)
                end
                loss[i] = sum(((x - goal).*weights).^2)
            end
            ind = sortperm(loss)[1:n_elite]
            μ_u = Statistics.mean(u[ind,:,:], dims=1)
            σ_u = Statistics.std(u[ind,:,:], dims=1)
            μ_t = Statistics.mean(t[ind], dims=1)
            σ_t = Statistics.std(t[ind], dims=1)
            
            temp_loss_min = min(loss...)
            if (loss_min - temp_loss_min)^2 < 1e-4
                count += 1
            end
            if temp_loss_min > 10 
                μ_u_0 = zeros(1, n_t, 4)
                μ_u_0[:,:,1] .= 10
                σ_u_0 = ones(1, n_t, 4)
                σ_u_0[1] *= 5

                μ_t_0 = 0.02
                σ_t_0 = 0.05
            end
            
            if temp_loss_min < loss_min
                loss_min = temp_loss_min
                best_u = u[ind[1],:, :]
                best_t = t[ind[1],:]
            end
            if verbose 
                println("ep loss:", temp_loss_min,"\tmin loss:",loss_min)
            end

            if  loss_min < 2e-1 || count > 5 
                break
            end
            
            
        end
    end
    i_t = 1
    for i = 1:length(best_t)
        if best_t[i] > 0
            i_t = i
            break
        end
    end

    return best_u[i_t,:], best_t[i_t], μ_u, σ_u, μ_t, σ_t
    
end
# end
# start = [-4.96396, -4.63075, 3.70707, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
# goal = [-4.76406, -3.89467, 3.40238, -0.0783476, -0.0292894, -0.0369011, 0.995812, -0.00658108, 0.0721031, 0.0179185, -0.0745848, -0.0625229, 0.0100022]

# μ_u_0 = zeros(1, 10, 4)
# μ_u_0[:,:,1] .= 10
# σ_u_0 = ones(1, 10, 4)
# σ_u_0[:,:,1] *= 5

# μ_t_0 = 0.02
# σ_t_0 = 0.05


# u, t, _, _, _, _ = cem(start, goal, ones(13), μ_u_0, σ_u_0, μ_t_0, σ_t_0)
# println(u,"\t", t)