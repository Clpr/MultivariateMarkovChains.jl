#===============================================================================
APPROXIMATION/DISCRETIZATION METHODS

1. From AR(1) to MultivariateMarkovChain
    - Tauchen (1986)
    - Rouwenhorst (1995)
2. From continuous state mappings to MultivariateMarkovChain
    - Young (2010) non-stochastic simulation
    - Gridization + frequency counting
===============================================================================#
export tauchen
export young




# ==============================================================================
# AR(1) --> MultivariateMarkovChain
# ==============================================================================
"""
    tauchen(ar1::AR1, N::Int ; nσ::Real = 3)::MultivariateMarkovChain{1}

Discretizes an AR(1) process into a MultivariateMarkovChain with `N` states.
"""
function tauchen(ar1::AR1, N::Int ; nσ::Real = 3)::MultivariateMarkovChain{1}
    res = tauchen(N, ar1.ρ, ar1.σ, yMean = ar1.xavg, nσ = nσ)
    return MultivariateMarkovChain(
        [[x,] for x in res.states],
        res.probs,
        validate  = true,
        normalize = true,
    )
end # tauchen















# ==============================================================================
# Young (2010) non-stochastic simulation
# ==============================================================================
"""
    young(
        f2fit ::Function,
        grids ::Vector{<:AbstractVector};
    )::MultivariateMarkovChain

Discretizes a continuous state mapping `f2fit` over the Cartesian product of 
`grids`, using Young (2010) non-stochastic simulation method.

The `f2fit` function should take a vector of the same dimension as the number of
grids and return a vector of the same dimension. The grids should be defined for
each dimension of the state space.

The function returns a `MultivariateMarkovChain` with the states and the
transition probabilities.

Note: the mapping should map the state space to itself. All outsider predictions
will be mapped to the nearest grid point.

## Reference

Young, Eric R. “Solving the Incomplete Markets Model with Aggregate Uncertainty 
Using the Krusell–Smith Algorithm and Non-Stochastic Simulations.” Journal of 
Economic Dynamics and Control 34, no. 1 (2010): 36–41. 

https://doi.org/10.1016/j.jedc.2008.11.010.
"""
function young(
    f2fit ::Function,
    grids ::Vector{<:AbstractVector};
)::MultivariateMarkovChain

    # step 1: construct the Cartesian space
    Xthis = Iterators.product(grids...) |> collect .|> collect |> vec

    # step 2: eval the operator for every grid point
    Xnext = [f2fit(xy) for xy in Xthis]

    # step 3: gridize/snap the next states to the grids
    Xnext_gridized = gridize(Xnext, grids)[1]

    # step 4: count the frequency of state transitions
    freqs = DataStructures.counter(Xthis .=> Xnext_gridized)

    # step 5: construct the transition matrix
    Pr = Float64[
        freqs[xi => xj]
        for (xi,xj) in Iterators.product(Xthis,Xthis)
    ]
    Pr ./= sum(Pr, dims = 2) # row-wise normalize

    return MultivariateMarkovChain(
        Xthis,
        Pr,
        validate  = true,
        normalize = true,
    )
end # young



































