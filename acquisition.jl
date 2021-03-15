include("shared.jl")
include("common_model.jl")
using GenParticleFilters
using Plots
gr()
Plots.GRBackend()


@gen function model(xs::Vector{Float64})
    n = length(xs)

    # sample covariance function
    covariance_fn::Node = @trace(covariance_prior(1), :tree)

    # sample diagonal noise
    noise = @trace(gamma(1, 1), :noise) + 0.01

    # compute covariance matrix
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)

    # sample from multivariate normal
    ys = Float64[]
    mu = 0
    var = cov_matrix[1,1]
    covm_22_inv = [1/cov_matrix[1,1]]
    for (i,x) in enumerate(xs)
        # condition on previous ys
        if i > 1
            covm_11 = cov_matrix[i,i]
            covm_21 = cov_matrix[1:i-1,i]
            covm_12 = transpose(covm_21)
            if i > 2
                covm_22_inv = blockwise_inv(cov_matrix, covm_22_inv, i)
            end
            mu = (covm_12 * covm_22_inv * ys)[1]
            var = covm_11[1] - (covm_12 * covm_22_inv * covm_21)[1]
        end
        y = {(:y, i)} ~ normal(mu, sqrt(var))
        push!(ys, y)
    end
    return (covariance_fn, ys)
end

function particle_filter(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj, x_obs_traj, y_obs_traj)
    # n_obs = length(xs)
    n_obs = 70
    obs_idx = [1]
    obs_xs = [xs[1]]
    obs_ys = [ys[1]]
    obs_choices = [choicemap(((:y, 1), ys[1]))]
    potential_xs = deepcopy(xs)
    deleteat!(potential_xs, 1)


    state = pf_initialize(model, ([xs[1]],), obs_choices[1], n_particles)
    # Iterate across timesteps
    for t=1:n_obs-1
        # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < 0.5 * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end

        # select next observation point
        next_obs = get_next_obs_x(state, potential_xs, obs_xs, obs_ys)
        push!(obs_idx, next_obs)
        push!(obs_xs, xs[next_obs])
        push!(obs_ys, ys[next_obs])
        deleteat!(potential_xs, next_obs)

        # Update filter state with new observation at timestep t
        push!(obs_choices, choicemap(((:y, t+1), ys[next_obs])))
        pf_update!(state, (obs_xs,), (UnknownChange(),), obs_choices[t+1])
        push!(x_obs_traj, xs[next_obs])
        push!(y_obs_traj, ys[next_obs])
        if mod(t,5) == 0
            println(obs_idx)
            println("number of observations: $t")
            callback(state, xs, ys, anim_traj, t)
        end
    end
    return state
end

# iterative
function get_next_obs_x(state, new_xs, x_obs, y_obs)
    k = 2
    e_ucb = zeros(Float64, length(new_xs))
    weights = get_norm_weights(state)

    for i=1:n_particles
        trace = state.traces[i]
        covariance_fn = get_retval(trace)[1]
        noise = trace[:noise]
        (conditional_mu, conditional_cov_matrix) = compute_predictive(
            covariance_fn, noise, x_obs, y_obs, new_xs)

        for j=1:length(new_xs)
            mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
            e_ucb[j] += (mu + k * var) * weights[i]
        end
    end
    return findmax(e_ucb)[2]
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

n_particles = 100
x_obs_traj = Float64[]
y_obs_traj = Float64[]
state = particle_filter(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj)

sorted_obs = []
for obs in keys(anim_traj)
    push!(sorted_obs, obs)
end

function make_animation()
    anim = @animate for obs in sort!(sorted_obs)
        vals = anim_traj[obs]
        obs_xs = x_obs_traj[1:obs]
        obs_ys = y_obs_traj[1:obs]
        pred_xs = xs

        inter_obs_x = Array{Float64,1}([obs_xs[1]])
        inter_obs_y = Array{Float64,1}([obs_ys[1]])
        obs_variances = Array{Float64,1}([0])
        for j=2:length(obs_xs)
            push!(inter_obs_x, (obs_xs[j]+obs_xs[j-1])/2)
            push!(inter_obs_y, (obs_ys[j]+obs_ys[j-1])/2)
            push!(obs_variances, 0)
        end

        # plot observations
        p = plot(xs, ys, title="$obs Observations, $n_particles Particles ", ylim=(-2, 3), legend=false, linecolor=:red)

        # get indices of the top n particles
        weights = [vals[i][3] for i=1:length(vals)]
        best_idxes = []
        for p=1:10
            max_weight_idx = findmax(weights)[2]
            push!(best_idxes, max_weight_idx)
            weights[max_weight_idx] = 0
        end

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
            # plot predictions for top 10 particles
            if i in best_idxes
                weights
                (conditional_mu, conditional_cov_matrix) = compute_predictive(
                    covariance_fn, noise, obs_xs, obs_ys, pred_xs)
                variances = []
                for j=1:length(pred_xs)
                    mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
                    push!(variances, sqrt(var)/mu)
                end
                pred_ys = mvnormal(conditional_mu, conditional_cov_matrix)
                plot!(p,pred_xs,pred_ys, linealpha = weight*15, ribbon=variances, fillalpha=weight*1.5)
            end
        end
        # plot!(p, inter_obs_x, inter_obs_y, ribbon=obs_variances,  fillalpha=0.3)
        plot!(p, obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 3, 0.6, :green, stroke(1, 1, :black, :dot)))
    end

    gif(anim, "acquisition.gif", fps = 1)
end

make_animation()
