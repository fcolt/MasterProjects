using CSV
using DataFrames
using Distributions
using Gen

ratings_df = DataFrame(CSV.File("ratings.csv"))
movies_df = DataFrame(CSV.File("movies.csv"))

DataFrames.select!(ratings_df, Not(:timestamp)) #drop timestamp column as it's not needed

data = innerjoin(ratings_df, movies_df, on = :movieId) #join the two dataframes

DataFrames.select!(data, Not([:title, :genres])) #drop title,genre columns

data = unstack(data, :movieId, :rating) #create a pivot table (show each user grouped by each movie)

DataFrames.select!(data, Not(:userId)) #drop userId as each row is the userId

for col in eachcol(data)
    replace!(col, missing => 0)
end

ordered_columns = sort([parse(Int, x) for x in names(data)]) 

data = data[:, string.(ordered_columns)]
mapcols!(col -> 2 * col, data) # to transform scores from floats to ints (0 - 5 -> 0 - 10)
data = convert.(Int64, data) #convert column types from Vector{Union{Missing, Float64}} to Vector{Float64}

test_data = Matrix{Int64}(data[[19, 21, 475, 476, 477], [1, 2, 3, 4, 10]])

K = 5
no_users = size(test_data, 1)
no_items = size(test_data, 2)

@gen function hpf_model(
    K, 
    a_prime::Float64 = 0.3, b_prime::Float64 = 0.3, 
    c_prime::Float64 = 0.3, d_prime::Float64 = 0.3, 
    a::Float64 = 0.3, c::Float64 = 0.3
)
    
    xi = Float64[]
    for u = 1:no_users
        push!(xi, {(:xi, u)} ~ gamma(a_prime, a_prime/b_prime))
    end
    theta = Vector{Float64}[]
    for u = 1:no_users
        push!(theta, [{(:theta, u, k)} ~ gamma(a, xi[u]) for k = 1:K])
    end

    eta = Float64[]
    for i = 1:no_items
        push!(eta, {(:eta, i)} ~ gamma(c_prime, c_prime/d_prime))
    end
    beta = Vector{Float64}[]
    for i = 1:no_items
        push!(beta, [{(:beta, i, k)} ~ gamma(c, eta[i]) for k = 1:K])
    end

    y = Vector{Int64}[]
    for u = 1:no_users
        push!(y, [{(:y, u, i)} ~ poisson(transpose(theta[u]) * beta[i]) for i = 1:no_items])
    end
    
    y
end

function make_constraints(ratings::Matrix{Int64})
    constraints = Gen.choicemap()
    for u = 1:size(ratings, 1)
        for i = 1:size(ratings, 2)
            constraints[(:y, u, i)] = ratings[u,i]
        end
    end
    constraints
end

function block_resimulation_update(tr)
    latent_variable = Gen.select(:xi)
    (tr, _) = mh(tr, latent_variable)

    latent_variable = Gen.select(:theta)
    (tr, _) = mh(tr, latent_variable)

    latent_variable = Gen.select(:eta)
    (tr, _) = mh(tr, latent_variable)

    latent_variable = Gen.select(:beta)
    (tr, _) = mh(tr, latent_variable)
    
    tr

end

function block_resimulation_inference(K::Int64, ratings::Matrix{Int64}, n_burnin::Int64, n_samples::Int64)
    observations = make_constraints(ratings)
    (tr, _) = generate(hpf_model, (K,), observations)
    
    for iter = 1:n_burnin
        tr = block_resimulation_update(tr)
    end

    trs = []
    for iter = 1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
    end

    trs

end

n_iter = 90000
n_burnin = 50000

trs = block_resimulation_inference(K, test_data, n_burnin, n_iter)

theta_samples = [[[trs[iter][(:theta, u, k)] for k=1:K] for u = 1:no_users] for iter=1:n_iter]
beta_samples = [[[trs[iter][(:beta, i, k)] for k=1:K] for i = 1:no_items] for iter=1:n_iter]

function get_recommendations(theta_samples, beta_samples)
    theta_mean = zeros((no_users, K))
    beta_mean = zeros((no_items, K))
    
    for i = 1:n_iter
        theta_mean += hcat(theta_samples[i]...)
        beta_mean += hcat(beta_samples[i]...)
    end
    theta_mean = theta_mean / n_iter
    beta_mean = beta_mean / n_iter
    return transpose(theta_mean) * beta_mean
end

get_recommendations(theta_samples, beta_samples)

