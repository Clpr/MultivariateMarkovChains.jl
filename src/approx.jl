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

Approximate a controlled Markov process Y = (X,Z) where Z is totally exogenous
and X's transition depends on (X,Z) simultaneously, using Young (2010) 
non-stochastic simulation method.

Y' = (X',Z');
X' = f(X,Z);
Z' ~ MarkovChain(Z)

The function receivs a function `fxz(X,Z)` that receives two vectors of X and Z
respectively; a `Zproc` which is a `MultivariateMarkovChain` representing
the exogenous process Z; and a vector of grids `xgrids` for how to discretize X.

Returns a `MultivariateMarkovChain` representing the controlled process (X,Z).
The states are the Cartesian product of `xgrids` and the states of `Zproc`.
The transition probabilities are computed based on the mapping `fxz` applied to
the states of (X,Z).

The returned `MultivariateMarkovChain` has a sparse transition matrix as the 
Young's algorithm considers only the first nearest neighbors.

## Tips

- Constructing the transition matrix requires evaluating `fxz` for many times.
Make sure `fxz` is efficient, especially if the grids are large.

- `fxz` should allow `Z` be a `StaticVector`. This is for performance reasons,
as `Zproc` stores states as `StaticVector`s. Extra type conversions may slow
down the process.

## Example

```julia
import MultivariateMarkovChains as mmc

# define exognous process Z
Zproc = mmc.MultivariateMarkovChain(
    [[1,1],[2,1],[1,2],[2,2]],
    [
        0.81  0.09  0.09  0.01;
        0.09  0.81  0.01  0.09;
        0.09  0.01  0.81  0.09;
        0.01  0.09  0.09  0.81;
    ]
)

# define endogenous/controlled states X = (x1,x2), x1 ∈ [0,2], x2 ∈ [0,10]
xgrids = [
    LinRange(0,2,30),
    LinRange(0,10,30),
]

# define a mapping X' = f(X,Z); use sqrt for simplicity
fxz(X,Z) = begin
    x1p = clamp(sqrt(X[1]*Z[1]), 0, 2)
    x2p = clamp(sqrt(X[2]*Z[2]), 0, 10)
    return [x1p, x2p]
end

# Run Young's algorithm
mc_Y = mmc.young(fxz, Zproc, xgrids)

```
"""
function young(
    fxz   ::Function,
    Zproc ::MultivariateMarkovChain,
    xgrids::Vector{<:AbstractVector}
)::MultivariateMarkovChain

    #=--------------------------------------------------------------------------
    TECH NOTES:

    - the dimensions of `Zproc` is not necessarily Cartesian joined. Depends on
      how users model the exogenous process Z. ==> So this function cannot
      assume that the Zproc is Cartesian joined.
      
    - We do not check fxz's return types as it is so limited. but we pre-alloc
      the sized state space and try to assign (X',Z') to it. If `fxz` returns
      something that is incompatible with the state space, it will throw an
      error.

    - We use the knowledge about the ordering of: `Iterators.product(a,b)`

    [
        1      1;
        2      1;
        3      1;
        .      .;
        .      .;
        Na     1;
        1      2;
        2      2;
        3      2;
        .      .;
        .      .;
        Na-1  Nb;
        Na    Nb;
    ]

    To assign values to the transition matrix.

    - We require each grid in `xgrids` are unique and ascendingly sorted. This
      helps to well-define the nearest neighbor search.

    --------------------------------------------------------------------------=#

    # check: dimensionality
    Nx = length(xgrids)
    Nz = ndims(Zproc)
    @assert Nx > 0 "X grids must be non-empty."
    @assert Nz > 0 "Z process must be non-empty."

    # check: grid sizes
    Dx = xgrids .|> length |> prod
    Dz = length(Zproc)
    Dy = Dx * Dz
    @assert Dx > 0 "One or more dimension(s) of X has no grid points."
    @assert Dz > 0 "Z process must have at least one state."

    # check: x grids
    @assert all(issorted, xgrids) "All xgrids must be ascendingly sorted."
    @assert all(allunique, xgrids) "All xgrids must be unique."


    # step: malloc the markov chain for Y = (X,Z)
    Ystates = Vector{Vector{Float64}}(undef, Dy)
    PrY     = spzeros(Float64, Dy, Dy)

    # step: collect the state space for Y = (X,Z), and make the indexing mats
    Xtensor = Iterators.product(xgrids...)
    Ytensor = Iterators.product(Xtensor, Zproc.states)
    XZsub   = CartesianIndices((Dx,Dz))
    Xind    = LinearIndices(ntuple(
        i -> length(xgrids[i]),
        length(xgrids)
    ))
    

    # step: fill the malloc-ed
    # HINTS: we need the location of `X` in `Xtensor` to compute the probs
    # HINTS: iy::Int, X::NTuple{Nx,Float64}, Z::StaticVector{Nz,Float64}
    for (iy,(X,Z)) in Ytensor |> enumerate

        # step: fill the state
        Ystates[iy] = Float64[X..., Z...]

        # step: evaluate X' = f(X,Z)
        Xp::Vector{Float64} = fxz(X |> collect, Z) |> collect

        # step: locate (X,Z) in their own grids
        ix = XZsub[iy][1]
        iz = XZsub[iy][2]

        # step: find neighbor grid points for X'; compute normalized distances
        Xp_neighbors_sub = neighbors_2combination(Xp, xgrids)
        Xp_neighbors = [
            getindex.(xgrids, sub)
            for sub in Xp_neighbors_sub
        ]
        ΔXp_neighbors = [
            normalized_distance(Xp, x, last.(xgrids) .- first.(xgrids)) 
            for x in Xp_neighbors
        ]
        
        # step: build conditional distribution
        # Pr{X'|X,Z,Z'}, same across all Z' states; length is Dx
        PrXp_givenXZZp = if length(Xp_neighbors) == 0
            # (X,Z) is exactly on-grid, no neighbors, so the density concentrate
            sparsevec(Int[ix], Float64[1.0], Dx)
        else
            pr = sparsevec(
                Int[
                    Xind[xsub...]
                    for xsub in Xp_neighbors_sub
                ],
                ΔXp_neighbors,
                Dx
            )
            pr ./ sum(pr) # normalize
        end

        # step: fill the corresponding columns in the `iy`-th row of PrY
        # HINT: locate the columns by `iz`
        # HINT: Pr{X'|X,Z} = Pr{X'|X,Z,Z'} * Pr{Z'|Z}
        # HINT: use Kronecker/tensor product to broadcast the probabilities
        PrY[iy,:] = kron(
            Zproc.Pr[iz,:],
            PrXp_givenXZZp
        )

    end # (iy,(X,Z))

    return MultivariateMarkovChain(
        Ystates,
        PrY,
        validate  = true,
        normalize = false,
        sparsify  = true,
    )
end # young






















