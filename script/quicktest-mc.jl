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

# default setup
mc = mmc.MultivariateMarkovChain(states, Pr) 

# check if `Pr` is really a transition matrix
mc = mmc.MultivariateMarkovChain(states, Pr, validate = true)

# rowwise-normalize `Pr` to ensure it is a valid transition matrix
mc = mmc.MultivariateMarkovChain(states, rand(2,2), normalize = true)


# the transition matrix can be sparse for many-state Markov chains
mmc.MultivariateMarkovChain(
    states,
    mmc.sparse([
        1.0 0.0;
        0.0 1.0
    ]),
)

# or, sparsify the transition matrix
mc = mmc.MultivariateMarkovChain(
    states, 
    [
        1.0 0.0;
        0.0 1.0
    ], 
    sparsify = true,
)


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
unique(mc)



# ANALYSIS ---------------------------------------------------------------------

# solves the stationary distribution
ss = mmc.stationary(mc)

# computes the long-term mean of the states
Statistics.mean(mc)


