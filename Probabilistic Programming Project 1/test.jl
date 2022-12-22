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
no_users = size(test_data, 2)
no_items = size(test_data, 1)

@gen function hpf_model(
    K::Int64, 
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
    # xi = rand(Gamma(a_prime, a_prime/b_prime), no_users)
    # theta = [rand(Gamma(a, xi[u]), K) for u in range(1, no_users)]

    # eta = rand(Gamma(c_prime, c_prime/d_prime), no_items)
    # beta = [rand(Gamma(c, eta[i]), K) for i in range(1, no_items)]

    y = Vector{Int64}[]
    for u = 1:no_users
        push!(y, [{(:y, u, i)} ~ poisson(transpose(theta[u]) * beta[i]) for i = 1:no_items])
    end
    
    y
end

function make_constraints(ratings::Matrix{Int64})
    constraints = Gen.choicemap()
    for u = 1:size(ratings, 2)
        for i = 1:size(ratings, 1)
            constraints[(:y, u, i)] = ratings[u,i]
        end
    end
    constraints
end

function block_resimulation_update(tr)

    latent_variable = Gen.select(:theta)
    (tr, _) = mh(tr, latent_variable)

    latent_variable = Gen.select(:beta)
    (tr, _) = mh(tr, latent_variable)
    
    tr

end

function block_resimulation_inference(K::Int64, ratings::Matrix{Int64}, n_burnin::Int64, n_samples::Int64)
    # fix observed data
    observations = make_constraints(ratings)
    # generate feasible starting point
    (tr, _) = generate(hpf_model, (K,), observations)

    # throw to garbage (can be entirely irrelevant if the starting point is far away from the posterior)
    for iter = 1:n_burnin
        tr = block_resimulation_update(tr)
    end

    # start saving traces from here (for posterior restoration)
    trs = []
    for iter = 1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
    end

    trs

end

trs = block_resimulation_inference(5, test_data, 0, 10000)