include("shared.jl")
include("common_model.jl")
using GenParticleFilters
using Plots
gr()
Plots.GRBackend()

function particle_filter(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj)
    n_obs = length(xs)
    obs_choices = [choicemap(((:y, t), ys[t])) for t=1:n_obs]
    state = pf_initialize(model, ([xs[1]],), obs_choices[1], n_particles)
    # Iterate across timesteps
    for t=2:n_obs
        # # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < 0.5 * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end
        # Update filter state with new observation at timestep t
        println("t", string(t))
        pf_update!(state, (xs[1:t],), (UnknownChange(),), obs_choices[t])
        if mod(t,10) == 0
            println("number of observations: $t")
            callback(state, xs, ys, anim_traj, t)
        end
    end
    return state
end

# load and rescale the airline dataset
(xs, ys) = get_airline_dataset()
xs_train = xs[1:100]
ys_train = ys[1:100]
xs_test = xs[101:end]
ys_test = ys[101:end]

# visualization
anim_traj = Dict()

# set seed
Random.seed!(1)

pf_callback = (state, xs, ys, anim_traj, t) -> begin
    # calculate E[MSE]
    n_particles = length(state.traces)
    e_mse = 0
    e_pred_ll = 0
    weights = get_norm_weights(state)
    if haskey(anim_traj, t) == false
        push!(anim_traj, t => [])
    end
    for i=1:n_particles
        trace = state.traces[i]
        covariance_fn = get_retval(trace)[1]
        noise = trace[:noise]
        push!(anim_traj[t], [covariance_fn, noise, weights[i]])
        mse =  compute_mse(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        pred_ll = predictive_ll(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        e_mse += mse * weights[i]
        e_pred_ll += pred_ll * weights[i]
    end
    println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
end

# do inference, time it
# @time (covariance_fn, noise) = inference(xs_train, ys_train, 1000, callback)
n_particles = 100
state = particle_filter(xs_train, ys_train, n_particles, pf_callback, anim_traj)

# visualization
# (conditional_mu, conditional_cov_matrix) = compute_predictive(
#     covariance_fn, noise, xs, ys, new_xs)

sorted_obs = []
for obs in keys(anim_traj)
    push!(sorted_obs, obs)
end
anim = @animate for obs in sort!(sorted_obs)
    vals = anim_traj[obs]
    obs_xs = xs_train[1:obs]
    obs_ys = ys_train[1:obs]
    pred_xs = xs[obs+1:length(xs)]

    inter_obs_x = Array{Float64,1}([obs_xs[1]])
    inter_obs_y = Array{Float64,1}([obs_ys[1]])
    obs_variances = Array{Float64,1}([0])
    for j=2:length(obs_xs)
        push!(inter_obs_x, (obs_xs[j]+obs_xs[j-1])/2)
        push!(inter_obs_y, (obs_ys[j]+obs_ys[j-1])/2)
        push!(obs_variances, 0)
    end

    # plot observations
    p = plot(obs_xs, obs_ys, title="$obs Observations, $n_particles Particles ", ylim=(-2, 3), legend=false, linecolor=:red)

    # plot predictions
    for i=1:length(vals)
        covariance_fn = vals[i][1]
        noise = vals[i][2]
        weight = vals[i][3]
        # calculate variance on observed data
        (obs_conditional_mu, obs_conditional_cov_matrix) = compute_predictive(
            covariance_fn, noise, obs_xs, obs_ys, inter_obs_x)
        for j=1:length(inter_obs_x)
            mu, var = obs_conditional_mu[j], obs_conditional_cov_matrix[j,j]
            obs_variances[j] += sqrt(var)/mu * weight
        end
        # plot predictions for every 5th particle
        if mod(i,5) ==0
            (conditional_mu, conditional_cov_matrix) = compute_predictive(
                covariance_fn, noise, obs_xs, obs_ys, pred_xs)
            variances = []
            for j=1:length(pred_xs)
                mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
                push!(variances, sqrt(var)/mu)
            end
            pred_ys = mvnormal(conditional_mu, conditional_cov_matrix)
            plot!(p,pred_xs,pred_ys, linealpha = weight*7, ribbon=variances, fillalpha=weight*3)
        end
    end
    plot!(p, inter_obs_x, inter_obs_y, ribbon=obs_variances,  fillalpha=0.3)
    plot!(p, obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 3, 0.6, :orange, stroke(1, 1, :black, :dot)))
end

gif(anim, "pf_version.gif", fps = 1)
