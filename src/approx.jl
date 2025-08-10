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
export young2
export young3




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
    young2(
        fxz   ::Function,
        Zproc ::MultivariateMarkovChain,
        xgrids::Vector{<:AbstractVector}
    )::MultivariateMarkovChain

Approximate a controlled Markov process Y = (X,Z) where Z is totally exogenous
and X's transition depends on (X,Z) simultaneously, using Young (2010) 
non-stochastic simulation method.

```plain
Y' = (X',Z');
X' = f(X,Z);
Z' ~ MarkovChain(Z)
```

This kind of controlled Markov processes are common in economics, esp. models
with only one (type of) agents. (e.g. Krusell-Smith)

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
function young2(
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













# ------------------------------------------------------------------------------
"""
    young3(
        f     ::Function,
        xgrids::Vector{<:AbstractVector},
        qgrids::Vector{<:AbstractVector},
        Zproc ::MultivariateMarkovChain,
    )::MultivariateMarkovChain

Approximate a controlled Markov process Y = (X,Q,Z) where Z is totally exogenous
, Q is a contemporary function of (X,Z), and X's transition depends on (X,Q,Z) 
simultaneously, using Young (2010) non-stochastic simulation method.

```plain
Y  := (X,Q,Z);
Q  := g(X,Z);
X' := h(X,Q,Z);
Z' ~ MarkovChain(Z);
stochastic function: (X',Q') = f(X,Q,Z,Z');

==> controlled Markov chain: Y
```

This kind of controlled Markov processes are common in economics, esp. models
with two or more (types of) agents in which `Q` is typically price functions,
optimal policy functions and other equilibrium profiles that implicitly involve
multi agent's optimality conditions. (e.g. TANK, THANK)

Receives:

- a function `f(X,Q,Z,Zp)` that receives three vectors of X, Q, Z, and a
given (future) Zp respectively. The function should returns two vectors: Xp,Qp
which represents the possible states of X and Q after transiting to the next
period.
- a `Zproc` which is a `MultivariateMarkovChain` representing the exogenous 
process Z.
- a vector of grids `xgrids` for how to discretize X.
- a vector of grids `qgrids` for how to discretize Q.

Returns a `MultivariateMarkovChain` representing the controlled process 
Y = (X,Q,Z).
The states are the Cartesian product of `xgrids`, `qgrids`, and the states of 
`Zproc`.
The transition probabilities are computed based on the mapping `f` applied to
the states of (X,Q,Z).

The returned `MultivariateMarkovChain` has a sparse transition matrix as the 
Young's algorithm considers only the first nearest neighbors.

## Notes:

- the dimensions of `Zproc` is not necessarily Cartesian joined. Depends on
how users model the exogenous process Z. ==> So this function cannot assume that
the Zproc is Cartesian joined.
- We do not check `f` return types as it is so limited. but we pre-alloc the 
sized state space and try to assign (X',Q',Z'). If `fxz` returns something that 
is incompatible with the state space, it will throw an error.
- We require each grid in `xgrids` and `qgrids` are unique and ascendingly 
sorted. This helps to well-define the nearest neighbor search.
- The algorithm evaluates `f` for numerous times. Make sure it is efficient.
Parallerization is planned. Will add later
"""
function young3(
    f     ::Function,
    xgrids::Vector{<:AbstractVector},
    qgrids::Vector{<:AbstractVector},
    Zproc ::MultivariateMarkovChain ;
    parallel::Bool = false,
)::MultivariateMarkovChain

    #=--------------------------------------------------------------------------
    TECH NOTES:

    - the dimensions of `Zproc` is not necessarily Cartesian joined. Depends on
      how users model the exogenous process Z. ==> So this function cannot
      assume that the Zproc is Cartesian joined.
      
    - We do not check f's return types as it is so limited. but we pre-allocate
      the sized state space and try to assign (X',Z') to it. If `f` returns
      something that is incompatible with the state space, it will throw an
      error.

    - We require each grid in `xgrids` and `qgrids` are unique and ascendingly 
      sorted. This helps to well-define the nearest neighbor search.

    --------------------------------------------------------------------------=#

    # validation
    @assert length(xgrids) > 0 "At least one dimension grid for X"
    @assert length(qgrids) > 0 "At least one dimension grid for Q"
    @assert all(
        isvalid_grid, 
        xgrids
    ) "X grids must be non-empty, ascendingly sorted and unique"
    @assert all(
        isvalid_grid, 
        qgrids
    ) "Q grids must be non-empty, ascendingly sorted and unique"

    # meta information
    #=--------------------------------------------------------------------------
    note: 2 types of index info:
    1. a scalar element's index in a vector-valued state e.g. X,Q,Z
    2. a vector-valued state in a grid
    We need both to correctly construct the transition matrix.

    I use small `x`, `q`, `z` to denote the dimensions of every X,Q,Z state;
    and use big `X`, `Q`, `Z` to denote the vector-valued state.
    --------------------------------------------------------------------------=#
    # dimensionality and grid size
    DimX = length(xgrids); DimQ = length(qgrids)

    # indexings: elements in X and Q
    subsx = CartesianIndices(xgrids .|> length |> Tuple) # DimX-D
    subsq = CartesianIndices(qgrids .|> length |> Tuple) # DimQ-D

    # grid sizes
    NX = subsx |> length
    NQ = subsq |> length
    NZ = length(Zproc)

    # indexings: Y = (X,Q,Z) as vector-valued states
    subsXQZ = CartesianIndices((NX,NQ,NZ)) # 3-D
    indsXQZ = LinearIndices(subsXQZ)

    # malloc
    #=--------------------------------------------------------------------------
    note: 
    - Use IJV to construct the large sparse transition matrix in the very end,
      Updating the transition matrix is the absolute performance bottleneck.
    - But the states can be constructed early as it does not change.
    - The IJV malloc-ed should accommodate the multi-threading.
    - Every thread writes to its own IJV subvectors, so simple thread-vectors
      work. And because IJV by definition locates everything in the transition
      matrix, just append new subvectors.
    --------------------------------------------------------------------------=#
    # transition matrices; for multi-threading
    PrIs = Vector{Int}[Int[] for _ in 1:Threads.nthreads()]
    PrJs = Vector{Int}[Int[] for _ in 1:Threads.nthreads()]
    PrVs = Vector{Float64}[Float64[] for _ in 1:Threads.nthreads()]

    # (X,Q) as one vector; used in computing the conditional distributions
    xqAll = Iterators.product(xgrids...,qgrids...) |> collect

    # (X,Q) joint grids; used for computing the conditional distributions
    xqgrids  = [xgrids;qgrids]
    Δxqgrids = last.(xqgrids) .- first.(xqgrids)
    indsxq   = LinearIndices((xqgrids .|> length) |> Tuple)
    indsXQ   = LinearIndices((NX,NQ))
    
    # Y = (X,Q,Z) states; stacked
    YAll = hcat(
        kron(ones(NZ), stack(xqAll |> vec, dims = 1)),
        kron(stack(Zproc.states, dims = 1), ones(NX * NQ))
    )

    # step: fill the malloc-ed sparse transition matrix
    # HINTS: we need the location of a (X,Q) point in the tensor space of (X,Q)
    #        to compute the probs.
    # HINTS: iy::Int, X, Q::NTuple{Nx,Float64}, Z::StaticVector{Nz,Float64}
    @maybe_threads parallel for subXQZ in subsXQZ

        tid = Threads.threadid()

        # which row in the transition matrix?
        iX = subXQZ[1]
        iQ = subXQZ[2]
        iZ = subXQZ[3]
        indXQZ = indsXQZ[subXQZ]

        # which (X,Q) joint vector (DimX+DimQ dims) in the grid?
        xq    = YAll[indXQZ,1:(DimX+DimQ)]
        indXQ = indsXQ[iX,iQ]

        # step: for every possible Z' state, eval (X',Q') = f(X,Q,Z,Z'), make
        #       conditional distributions, then fill the transition matrix.
        for jZp in 1:NZ

            # eval: (X',Q') = f(X,Q,Z,Zp)
            xp,qp = f(
                xq[1:DimX],
                xq[DimX+1:end],
                Zproc.states[iZ],
                Zproc.states[jZp],
            )
            xpqp = [xp;qp] # length: DimX + DimQ

            # find neighbor grid points for (X',Q') as vector of sub tuples
            subsXpQpNeighbors = neighbors_2combination(xpqp, xqgrids)

            # materialize the neighbor points for computing the distances
            XpQpNeighbors = Vector{Float64}[
                getindex.(xqgrids,subAsTup) 
                for subAsTup in subsXpQpNeighbors
            ]

            # compute the normalized distances
            Δs = Float64[
                normalized_distance(xpqp, xpqp_candi, Δxqgrids)
                for xpqp_candi in XpQpNeighbors
            ]

            # build conditional distribution Pr{X',Q'|X,Q,Z,Z'}
            # note: the column locations `JsCondi` are in the (X,Q) grid
            JsCondi, VsCondi = if length(XpQpNeighbors) == 0
                # case: (X',Q') is exactly on the grid; no neighbors so the
                #       density concentrates to the point itself
                Int[indXQ,], Float64[1.0,]
            else
                # case: (X',Q') is not on the grid; the prob split to multiple
                #       grid points; the density is proportional to the distance
                #       between the neighbor grid points and (X',Q').
                # hint: needs to convert ind of (X,Q) to ind (x1,x2,...q1,q2...)
                _Js = Int[
                    indsxq[CartesianIndex(subxq)]
                    for subxq in subsXpQpNeighbors
                ]

                _Js, Δs ./ sum(Δs)
            end # if

            # convert the conditional distributions to the unconditional ones
            JsCondi .+= indsXQZ[1,1,jZp] - 1
            VsCondi .*= Zproc.Pr[iZ,jZp]

            # push
            append!(PrIs[tid], fill(indXQZ, JsCondi |> length))
            append!(PrJs[tid], JsCondi)
            append!(PrVs[tid], VsCondi)

        end # jZp
    end # subXQZ

    # build the sparse transition matrix
    PrY = sparse(
        reduce(vcat,PrIs),
        reduce(vcat,PrJs),
        reduce(vcat,PrVs),
        size(YAll,1),
        size(YAll,1),
    )

    return MultivariateMarkovChain(
        YAll |> eachrow,
        PrY,
        validate  = true,
        normalize = false,
        sparsify  = true,
    )
end # young3




