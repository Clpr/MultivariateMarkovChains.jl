#===============================================================================
MULTIVARIATE MARKOV CHAIN REPRESENTATIONS

1. MultivariateMarkovChain

## overloads
- Base.rand()
- Base.merge()
===============================================================================#
export MultivariateMarkovChain


# ==============================================================================
# CLASS DEFINITIONS
# ==============================================================================
"""
    MultivariateMarkovChain{D}

A structure to represent a multivariate Markov chain with discrete states and 
transition probabilities. A state is a numeric vector of dimension `D`, and the 
transition matrix is a square matrix where each entry represents the probability
of transitioning from one state (row) to another (column).


## Example
```julia
import Statistics
import MultivariateMarkovChains as mmc


# INITIALIZATION ---------------------------------------------------------------

# define a multivariate Markov chain with 3 states in 2D
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
unique(mc)

# split the multivariate Markov chain into `D` independent univariate chains
# overloads `Base.split`
mc_univariate = split(mc)



# ANALYSIS ---------------------------------------------------------------------

# solves the stationary distribution
ss = stationary(mc)

# computes the long-term mean of the states
Statistics.mean(mc)

```
"""
mutable struct MultivariateMarkovChain{D}

    # number of states (N)
    N::Int

    # states, as a N-vector of D-dim numeric vectors
    states::Vector{StaticVector{D,Float64}}

    # transition matrix, as a N x N matrix
    Pr::Matrix{Float64}

    function MultivariateMarkovChain(
        states   ::AbstractVector, 
        Pr       ::Matrix{Float64} ;
        validate ::Bool = true,
        normalize::Bool = true,
    )
        N = length(states)
        D = length(states[1])

        @assert N > 0 "There must be at least one state"
        @assert issquare(Pr) "Transition matrix must be square"
        @assert (size(Pr, 1) == N) "Transition matrix must match the states"
        @assert all(length.(states) .== D) "All states must have the same dim"
        
        if validate
            flagValid = all(isapprox.(sum(Pr, dims=2), 1.0, atol = 1E-4))
            if !flagValid
                @warn "Each row of the transition matrix must sum to 1.0"
            end
            if (!flagValid) && normalize
                @warn "Normalizing transition matrix to make row sums = 1.0"
                Pr .= Pr ./ sum(Pr, dims=2)
            end
        end # if
        new{D}(
            N,
            states .|> SVector{D,Float64},
            Pr
        )
    end
end # MultivariateMarkovChain{D}
# ------------------------------------------------------------------------------
function Base.show(io::IO, mc::MultivariateMarkovChain{D}) where {D}
    println(io, "MultivariateMarkovChain with $(mc.N) states of dimension $D:")
    println(io, "  States: $(mc.N) x $D matrix")
    println(io, "  Transition matrix: $(mc.N) x $(mc.N) matrix")
end
# ------------------------------------------------------------------------------
function Base.ndims(mc::MultivariateMarkovChain{D}) where {D}
    return D
end # ndims
# ------------------------------------------------------------------------------
function Base.length(mc::MultivariateMarkovChain{D}) where {D}
    return mc.N
end # length
# ------------------------------------------------------------------------------
function Base.size(mc::MultivariateMarkovChain{D}) where {D}
    return (mc.N, mc.N)
end # size
# ------------------------------------------------------------------------------
"""
    stack(mc::MultivariateMarkovChain{D}) where {D}

Stack the states of the multivariate Markov chain into a single matrix of size
`D x N`, where `N` is the number of states. This function is useful for
visualizing or processing the states in a single matrix format.
"""
function Base.stack(mc::MultivariateMarkovChain{D}) where {D}
    return stack(mc.states)
end # stack
# ------------------------------------------------------------------------------
"""
    unique(mc::MultivariateMarkovChain{D}) where {D}

Collects every marginal state/dimension's unique values into vectors. Returns a 
`D`-tuple of `StaticVector{...,Float64}` where `...` is the number of unique
values in each dimension. This is useful for understanding the range of values
each state dimension can take across all states in the Markov chain.
"""
function Base.unique(mc::MultivariateMarkovChain{D}) where {D}
    return mc.states |> sac.invert .|> unique
end # unique
# ------------------------------------------------------------------------------
"""
    rand(mc::MultivariateMarkovChain{D}, T::Int; ...)

Simulates a Markov chain for `T` time steps, starting from the initial state
`x0`. Returns a matrix of size `D x T`, where each column represents the state
at a given time step. The first column is always the initial state `x0`, and
subsequent columns are generated based on the transition probabilities defined
in the Markov chain.
"""
function Base.rand(
    mc  ::MultivariateMarkovChain{D}, 
    T   ::Int ; 
    x0  ::AbstractVector     = mc.states[1],  # Initial state
    seed::Union{Nothing,Int} = nothing,
) where {D}

    x0s = x0 |> SVector{D,Float64}

    @assert x0s in mc.states "Initial state must be one of the defined states"

    # Locate the state index
    ix = findfirst(isequal(x0s), mc.states)

    xpath = Matrix{Float64}(undef, D, T)
    xpath[:, 1] = x0s

    ixpath    = Vector{Int}(undef, T)
    ixpath[1] = ix

    # Draw random numbers for state transitions
    shocks = Base.rand(MersenneTwister(seed), T-1)
    
    for t in 2:T
        ix = inv_discrete_distribution(
            shocks[t-1],
            mc.Pr[ix, :],
        )
        xpath[:, t] = mc.states[ix]
        ixpath[t]   = ix
    end # t
    return xpath, ixpath
end # rand
# ------------------------------------------------------------------------------
"""
    merge(mc1::MultivariateMarkovChain{D}, mc2::MultivariateMarkovChain{M})

Merges two independent Markov chains `mc1` and `mc2` into a single Markov chain
of dimension `D + M`. The states of the new chain are formed by Cartesian
product of the states of `mc1` and `mc2`. The values are joined into one vector
e.g. [x1,] merge [y1,y2] --> [x1,y1,y2] where x1,y1,y2 are the dimensional
elements of the two Markov chains respectively.
"""
function Base.merge(
    mc1::MultivariateMarkovChain{D}, 
    mc2::MultivariateMarkovChain{M}
) where {D,M}
    N2     = mc1.N * mc2.N
    states = Vector{Vector{Float64}}(undef, N2)
    Pr     = Matrix{Float64}(undef, N2, N2)

    linidx = LinearIndices((1:mc1.N, 1:mc2.N))

    for ix in 1:mc1.N, iy in 1:mc2.N
        x    = mc1.states[ix]
        y    = mc2.states[iy]
        irow = linidx[ix,iy]

        states[irow] = Float64[x; y]

        for jx in 1:mc1.N, jy in 1:mc2.N
            jcol = linidx[jx,jy]
            Pr[irow, jcol] = mc1.Pr[ix, jx] * mc2.Pr[iy, jy]
        end # (jx,jy)
    end # (ix,iy)

    return MultivariateMarkovChain(
        states, Pr,
        validate  = true,
        normalize = true,
    )
end # merge
# ------------------------------------------------------------------------------
"""
    stationary(mc::MultivariateMarkovChain{D}) where {D}

Computes the stationary distribution of a multivariate Markov chain. Returns a 
vector of probabilities representing the long-term distribution of states in the
chain. The function uses the eigenvalue decomposition of the transition matrix
to find the stationary distribution.

Alternatively, one can always use a recursive approach (i.e. `mc.Pr^n`) to 
compute the stationary distribution where `n` is large enough.
"""
function stationary(mc::MultivariateMarkovChain{D}) where {D}
    return stationary(mc.Pr)
end # stationary
# ------------------------------------------------------------------------------
"""
    mean(mc::MultivariateMarkovChain{D}) where {D}

Computes the mean of the states in a multivariate Markov chain, weighted by the
stationary distribution. Returns a `Vector{Float64}` of length `D`, where each
element is the mean of the corresponding state dimension.
"""
function Statistics.mean(mc::MultivariateMarkovChain{D}) where {D}
    ss  = stationary(mc)
    res = zeros(D)
    for state in mc.states
        res .+= state .* ss
    end
    return res
end # mean







