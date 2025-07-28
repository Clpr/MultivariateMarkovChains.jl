mmc = include("../src/MultivariateMarkovChains.jl")
# import MultivariateMarkovChains as mmc


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tauchen (1986)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ar1 = mmc.AR1(ρ = 0.9, xavg = 10.0, σ = 0.1)

# style 1: OOP, returns a `MultivariateMarkovChain`
mmc.tauchen(
    ar1,
    5, # number of discretized states
    nσ = 2.5, # spanning how many standard deviation around the mean
)

# style 2: generic, returns a NamedTuple of `states` and `probs`
mmc.tauchen(
    5, # number of discretized states
    ar1.ρ, # persistence
    ar1.σ, # standard deviation
    yMean = ar1.xavg, # mean
    nσ = 2.5, # spanning how many standard deviation around the mean
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Young (2010) non-stochastic simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# an example 2-D operator mapping continuous vector states to the same space
# e.g. consider [0,1]^2 space for simplicity
# e.g. do flipping around (0.5,0.5) for example
f2fit(X) = clamp.([1.0 - X[1], 1.0 - X[2]], 0.0, 1.0)

# design grids for discretizing the continuous state space
grids = [
    LinRange(0,1,11), # 11 points in the first dimension
    LinRange(0,1,21), # 21 points in the second dimension
]

# style 1: let the package do the work
# returns a `MultivariateMarkovChain{2}` with 11*21 = 231 states
mc = mmc.young(
    f2fit, # the operator mapping
    grids, # the grids for discretization
)

# style 2: break down the steps and control more details
# returns a NamedTuple of `states` and `probs` which can be used to construct 
# a `MultivariateMarkovChain`

# step 2.1: evaluate the operator for every grid point, Cartesian/tensor joined
# and stack the results for convenience
Xthis = Iterators.product(grids...) |> collect .|> collect |> vec
Xnext = [f2fit(xy) for xy in Xthis] |> stack

# step 2.2: gridize/snap the next states to the grids; each dimension has its own grid
Xnext_gridized = [
    [x1,x2]
    for (x1,x2) in zip(
        grids[1][mmc.gridize(Xnext[1,:], grids[1])],
        grids[2][mmc.gridize(Xnext[2,:], grids[2])],
    )
]

# step 2.3: count the frequency of state transitions
# tips: use DataStructures.Counter
freqs = mmc.DataStructures.counter(Xthis .=> Xnext_gridized)

# step 2.4: construct the transition probability matrix
Pr = Float64[
    freqs[xi => xj]
    for (xi,xj) in Iterators.product(Xthis,Xthis)
]
# Pr |> mmc.sparse # (optional) check the sparsity pattern
Pr ./= sum(Pr, dims = 2) # normalize the rows to sum to 1


# step 2.5: construct the `MultivariateMarkovChain`
mc = mmc.MultivariateMarkovChain(
    Xthis, # the states
    Pr,    # the transition probability matrix
    validate  = true, # validate the transition probabilities
    normalize = true, # normalize the rows to sum to 1
)

# Then, if the mapping `f2fit` is stochastic, one should merge the constructed
# `MultivariateMarkovChain` with the Markov chain of the stochastic states.



