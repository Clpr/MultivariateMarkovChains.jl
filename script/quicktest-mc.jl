import Statistics


# import MultivariateMarkovChains as mmc
mmc = include("../src/MultivariateMarkovChains.jl")




# INITIALIZATION ---------------------------------------------------------------

# define a multivariate Markov chain with 2 states in 2D
states = [
    [1.0, 2.0],
    [3.0, 4.0],
]
Pr = [
    0.8 0.2;
    0.1 0.9;
]

# check if `Pr` is really a transition matrix
mc = mmc.MultivariateMarkovChain(states, Pr, validate = true)

# rowwise-normalize `Pr` to ensure it is a valid transition matrix
mc = mmc.MultivariateMarkovChain(states, rand(2,2), normalize = true)

# default setup
mc = mmc.MultivariateMarkovChain(states, Pr) 


# ACCESS -----------------------------------------------------------------------

# states (as vector of static vectors)
mc.states

# transition matrix
mc.Pr


# META INFORMATION -------------------------------------------------------------

# dimensionality
ndims(mc) == 2

# number of states
length(mc)


# SIMULATION -------------------------------------------------------------------

# simulate a Markov chain for 10 time steps, starting from the 1st stored state
# overloads `Base.rand`
rand(mc, 10)

# simulate a Markov chain for 10 time steps, starting from a custom state
rand(mc, 10, x0 = [1.0, 2.0])
rand(mc, 10, x0 = mc.states[2])

# specify a random seed for reproducibility
rand(mc, 10, seed = 123)

# or, typical seed setting works as well
# Random.seed!(123)
rand(mc, 10)


# MANUPULATION -----------------------------------------------------------------

# merge two INDEPENDENT Markov chains into a single one of higher dimension
# overloads `Base.merge`
mc2 = merge(mc, mc)

# merge more than two Markov chains, the `reduce` function works naturally
mc3 = reduce(merge, [mc for _ in 1:6])

# stack the states into a single matrix of size `D x N`
stack(mc3)

# collect every state's unique values into vectors
# e.g. states [[1.0,3.0],[2.0,3.0]] --> [1.0,2.0] and [3.0,]
unique(mc3)

# notice: in this example, merging the splitted chains back will NOT yield the
# original multivariate Markov chain
merge(mc_univariate...) != mc


# ANALYSIS ---------------------------------------------------------------------

# solves the stationary distribution
ss = mmc.stationary(mc)

# computes the long-term mean of the states
Statistics.mean(mc)

# computes the long-term covariance matrix of the states
# TODO, not implemented yet
# Statistics.cov(mc)


# ESTIMATION -------------------------------------------------------------------
# Two available methods: 
#   1. OLS
#   2. Young (2010)'s deterministic simulation method
# 
# Two data formats:
#   1. single vector-valued time series
#   2. (today => tomorrow) pairs


ρ = fill(0.9,4) |> mmc.diagm
Σ = [
    1.0 0.1 0.1 0.1;
    0.1 1.0 0.1 0.1;
    0.1 0.1 1.0 0.1;
    0.1 0.1 0.1 1.0;
]
Xpath = let x0 = rand(4), T = 100
    xavg = zeros(4)
    path = zeros(4,T)

    ΦZ = mmc.dst.MvNormal(Σ)

    path[:,1] = zeros(4)
    for t in 2:T
        path[:,t] = ρ * path[:,t-1] + (mmc.I(4) - ρ) * xavg + rand(ΦZ)
    end

    path
end



# OLS, single vector-valued time series
mmc.fit(
    Xpath, # 1 time series of length-4 X(t) observations; of size D * T
    method = :ols
)

# OLS, (today => tomorrow) pairs
mmc.fit(
    Xpath[:,1:end-1], # 100 X(t) observations
    Xpath[:,2:end],   # and X(t+1) observations
    method = :ols
)
# or equivalent syntax
mmc.fit(
    [
        Xpath[:,i] => Xpath[:,i+1]
        for i in 1:size(Xpath, 2)-1
    ],
    method = :ols
)




#=

# Young (2010), single vector-valued time series
# all points are on-grid, or equivalently, discretize the time series as exactly
# what it has observed. There may be very many states if the data is de facto
# continuous.
mmc.fit(
    [rand(4) for _ in 1:100], # 1 time series of length-4 X(t) observations
    method = :young
)


=#











