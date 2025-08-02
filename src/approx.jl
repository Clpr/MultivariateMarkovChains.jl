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

Note: if you need to model a controlled Markov process (X,Z) where Z is totally
exogenous and X's transition depends on (X,Z) simultaneously, then you may want
to check out another `young(f,Zproc,xgrids)` API.

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
# ------------------------------------------------------------------------------
"""
    young(
        nextstates::Array{<:AbstractVector},
        grids::Vector{<:AbstractVector};
    )::MultivariateMarkovChain

Discretizes a continuous state mapping over the Cartesian product of `grids`, 
using Young (2010) non-stochastic simulation method.

Returns a `MultivariateMarkovChain{D}` where `D=length(grids)` is the dimension
of the state space. The `nextstates` is a `D`-dimensional array where each
element is a vector representing the next state for each grid point.

Note: if you need to model a controlled Markov process (X,Z) where Z is totally
exogenous and X's transition depends on (X,Z) simultaneously, then the current
API works if you manually provide the `nextstates` array properly. Otherwise,
you may want to check out another `young(f,Zproc,xgrids)` API for convenience.

## Notes
- If there are states in `nextstates` but not in the tensor/Cartesian grid 
constructed by `grids`, then they will be ignored.
"""
function young(
    nextstates::Array{<:AbstractVector},
    grids::Vector{<:AbstractVector};
)::MultivariateMarkovChain

    D = length(grids)
    @assert D > 0 "Grids must be non-empty."

    # step 1: construct the Cartesian space
    Xthis = Iterators.product(grids...) |> collect .|> collect |> vec

    gridSize = size(Xthis)
    @assert size(nextstates) == gridSize "Next states must match the grid size."

    # step 2: count the frequency of state transitions
    freqs = DataStructures.counter(Xthis .=> nextstates)

    # step 3: construct the transition matrix
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
# ------------------------------------------------------------------------------
"""
    young(
        fxz   ::Function,
        Zproc ::MultivariateMarkovChain,
        xgrids::Vector{<:AbstractVector}
    )::MultivariateMarkovChain

Approximate a controlled Markov process (X,Z) where Z is totally exogenous
and X's transition depends on (X,Z) simultaneously, using Young (2010) 
non-stochastic simulation method.

The function receivs a function `fxz(X,Z)` that receives two vectors of X and Z
respectively; a `Zproc` which is a `MultivariateMarkovChain` representing
the exogenous process Z; and a vector of grids `xgrids` for how to discretize X.

Returns a `MultivariateMarkovChain` representing the controlled process (X,Z).
The states are the Cartesian product of `xgrids` and the states of `Zproc`.
The transition probabilities are computed based on the mapping `fxz` applied to
the states of (X,Z).

"""
function young(
    fxz   ::Function,
    Zproc ::MultivariateMarkovChain,
    xgrids::Vector{<:AbstractVector}
)::MultivariateMarkovChain





end # young






















