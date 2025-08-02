#===============================================================================
MATH HELPERS

1. generic helpers
2. AR(1) discretization helpers
    - Tauchen (1986)
    - Rouwenhorst (1995)
3. Young (2010) deterministic simulation method
===============================================================================#
export remove_glitch!
export stationary
export gridize
export isstationary


# ------------------------------------------------------------------------------
function issquare(m::AbstractMatrix)::Bool
    # check if a matrix is square
    return size(m, 1) == size(m, 2)
end
# ------------------------------------------------------------------------------
"""
    ongrid(
        x    ::AbstractVector{<:Real}, 
        grids::Vector{<:AbstractVector{<:Real}}
    )::Bool

Check if a point `x` is on-grid with respect to the grids defined in `grids`.
Each vector in `grids` represents the grid points for the corresponding 
dimension of `x`. The function returns `true` if `x` is on-grid, meaning
that each element of `x` is exactly equal to one of the grid points in the
corresponding dimension of `grids`. Otherwise, it returns `false`.
"""
function ongrid(
    x    ::AbstractVector{<:Real}, 
    grids::Vector{<:AbstractVector{<:Real}}
)::Bool
    # check if a point x is on-grid with respect to the grids
    return all(x .∈ grids)
end
# ------------------------------------------------------------------------------
"""
    locate(x::Real, xsorted::AbstractVector) -> NTuple{2,Int}

Finds the indices of the two neighboring elements in the sorted vector `xsorted`
that bound the value `x`.

# Arguments
- `x::Real`: The value to locate within `xsorted`.
- `xsorted::AbstractVector`: A vector of values sorted in ascending order.

# Returns
- `NTuple{2,Int}`: A tuple `(i, j)` where:
    - If `x == xsorted[i]`, returns `(i, i)`.
    - Otherwise, returns `(i, i+1)` such that `xsorted[i] <= x < xsorted[i+1]`.

# Notes
- Assumes `x >= xsorted[1] && x <= xsorted[end]`; no validation is performed.
- Assumes `xsorted` is sorted in ascending order and unique.
- Useful for interpolation or binning tasks where the position of `x` relative 
to `xsorted` is needed.
"""
function locate(x::Real, xsorted::AbstractVector)::NTuple{2,Int}
    # find the two neighbor indices in xsorted such that
    # xsorted[i] <= x < xsorted[i+1]
    # return (i, i+1)
    # if x == xsorted[i], return (i, i)
    # xsorted: ascendingly
    # assume: x >= xsorted[1] && x <= xsorted[end]; no validation
    #=
    let xsorted = LinRange(0,1,10), x = 0.2
       println(xsorted)
       tmp = adp.locate(x,xsorted)
       println(tmp)
       println(xsorted[tmp[1]], "<=", x, "<=", xsorted[tmp[2]])
    end
    =#

    i1 = searchsortedlast(xsorted, x)
    i2 = searchsortedfirst(xsorted, x)

    return (i1, i2)
end
# ------------------------------------------------------------------------------
"""
    neighbors_2combination(
        x    ::AbstractVector{<:Real},
        grids::Vector{<:AbstractVector{<:Real}},
    )

Finds all the on-grid neighbors for D-dimensional point `x` given the grids
defined in `grids`. Each vector in `grids` represents the grid points for
the corresponding dimension of `x`.

The function returns a `::Vector{Tuple{Vararg{Int}}}`: a vector of tuples, where
each inner tuple represents an on-grid neighbor point. An inner tuple has total
D-elements that represents the indices of the nearest grid points in each
dimension of `x`.

The function can return at most 2^D neighbors (if no dimension of `x` locates at
a grid point of `grids` exactly); and can return at least 0 neighbor (if all
dimensions of `x` locate at a grid point of `grids` exactly, in other words, `x`
is already on-grid). If `x` is outside the range of `grids`, the function will
return 1 neighbor (the nearest boundary point in `grids`).

## Tips

- The function does not care about the value types of `x` and `grids`. To obtain
the values of the neighbors, you can use `getindex.(grids, ...)` to get the
values of the neighbors, where `...` is the indices returned by this function.
"""
function neighbors_2combination(
    x    ::AbstractVector{<:Real},
    grids::Vector{<:AbstractVector{<:Real}},
)::Vector{Tuple}

    @assert all(allunique, grids) "All grids must be unique"
    @assert all(issorted, grids) "All grids must be ascendingly sorted"

    if ongrid(x, grids)
        # x is already on-grid, no neighbors
        return Tuple{Vararg{Int}}[]
    end

    # bracket each element of x with the grids
    inds = Vector{Vector{Int}}(undef, length(x))
    for (i, xi) in enumerate(x)
        
        Ni = grids[i] |> length
        
        il,ir = locate(xi, grids[i])
        
        if il < 1
            # xi is smaller than the first grid point
            inds[i] = [1]
        elseif ir > Ni
            # xi is larger than the last grid point
            inds[i] = [Ni]
        else
            if il == ir
                # xi is exactly on a grid point
                inds[i] = [il]
            else
                # xi is between two grid points
                inds[i] = [il, ir]
            end
        end # if

    end

    return Iterators.product(inds...) |> collect |> vec
end
# ------------------------------------------------------------------------------
"""
    normalized_distance(
        x1     ::AbstractVector{<:Real},
        x2     ::AbstractVector{<:Real},
        xscales::AbstractVector{<:Real},
        degree ::Int = 2,
    )w

Computes the normalized distance between two D-dimensional points `x1` and `x2`,
where `xscales` is a vector of scaling factors for each dimension which usually
is the maximum-minimum range of the corresponding dimension.

The normalized distance is then defined as the relative distance between x1 and
x2 in a normalized space. The parameter `degree` specifies the degree of the
Minkowski distance metric to use, with the default being 2 (Euclidean distance).
"""
function normalized_distance(
    x1     ::AbstractVector{<:Real},
    x2     ::AbstractVector{<:Real},
    xscales::AbstractVector{<:Real} ;
    degree ::Int = 2,
)
    return norm( (x1 .- x2) ./ xscales, degree )
end
# ------------------------------------------------------------------------------
"""
    inv_discrete_distribution(pr::Real, probs::AbstractVector{Float64})::Int

Inverse discrete distribution function. Given a probability `pr` and a
discrete distribution `probs`, this function returns the index of the first
cumulative probability that is greater than or equal to `pr`. The `probs`
must be a vector of probabilities that sum to 1.0, and `pr` must be in the
range [0, 1].
"""
function inv_discrete_distribution(
    pr   ::Real, 
    probs::AbstractVector{Float64}
)::Int
    @assert (0.0 <= pr <= 1.0) "pr must be in the range [0, 1]"
    @assert isapprox(sum(probs), 1.0, atol=1e-4) "probs must sum to 1.0 at 1E-4"

    probs2    = probs ./ sum(probs)  # normalize to sum to 1.0
    cum_probs = cumsum(probs2)
    
    return findfirst(cum_probs .>= pr)
end # inv_discrete_distribution
# ------------------------------------------------------------------------------
"""
    remove_glitch!(mat::Matrix ; tol::Real = 1E-10)

Set all elements in `mat` that are smaller than `tol` in absolute value to zero.
"""
function remove_glitch!(mat::Matrix ; tol::Real = 1E-10)
    mat[abs.(mat) .< tol] .= 0.0
    return nothing
end # remove_glitch!
# ------------------------------------------------------------------------------
"""
    stationary(Pr::AbstractMatrix)::Vector{Float64}

Computes the stationary distribution of a Markov chain whose transition matrix
is `Pr`. The function uses the eigenvalue decomposition. The function returns
a vector of probabilities.
"""
function stationary(Pr::AbstractMatrix)::Vector{Float64}
    λ, V = eigen(Pr')
    i    = argmin(abs.(λ .- 1))
    pss  = V[:,i] ./ sum(V[:,i])
    return pss .|> real
end
# ------------------------------------------------------------------------------
"""
    gridize(
        Xs::AbstractVector{M}, 
        xGrid::AbstractVector{N}
    ) where {M<:Real, N<:Real}

Given a vector `Xs` and a grid `xGrid`, this function maps each element of `Xs`
to the nearest element in `xGrid`. The function returns a vector of the same
length as `Xs`, where each element is the index of the nearest element in
`xGrid` for the corresponding element in `Xs`. If an element in `Xs`
is outside the range of `xGrid`, it is mapped to the nearest boundary element
of `xGrid`.

The `xGrid` must be a unique vector but not necessarily sorted.
"""
function gridize(
    Xs   ::AbstractVector{M}, 
    xGrid::AbstractVector{N}
)::Vector{Int} where {N<:Real, M<:Real}
    @assert length(xGrid) > 0 "Grids must have at least one element"
    @assert all(unique(xGrid) .== xGrid) "All grids must be unique"

    return getindex.(argmin(abs.(Xs .- xGrid'), dims = 2) |> vec, 2)
end
# ------------------------------------------------------------------------------
"""
    gridize(
        Xs    ::Vector{<:AbstractVector},
        xGrids::Vector{<:AbstractVector},
    )::Tuple{Vector{Vector{Float64}},Vector{Vector{Int}}}

Given a vector of D-dimensional points stored in `Xs`, and a D-element vector 
of grids `xGrids`, the function maps each point in `Xs` to the nearest grid 
point in each dimension. The function returns two vectors of D-vectors, where 
each vector corresponds to the values of the nearest grid points in each 
dimension for the corresponding point in `Xs`, and the indices of these points.
"""
function gridize(
    Xs    ::Vector{<:AbstractVector},
    xGrids::Vector{<:AbstractVector},
)::Tuple{Vector{Vector{Float64}},Vector{Vector{Int}}}
    @assert length(Xs) > 0 "Xs must not be empty"
    @assert length(xGrids) > 0 "xGrids must not be empty"
    D = length(xGrids)
    @assert all(length.(Xs) .== D) "elements of Xs mismatch with the grids"
    @assert all(length.(xGrids) .> 0) "All grid must have at least one element"
    @assert all(unique.(xGrids) .== xGrids) "All grids must be unique"

    valNew = Vector{Vector{Float64}}(undef, length(Xs))
    idxNew = Vector{Vector{Int}}(undef, length(Xs))
    for (i, x) in enumerate(Xs)
        val = Vector{Float64}(undef, D)
        idx = Vector{Int}(undef, D)
        for d in 1:D
            res = (x[d] .- xGrids[d]) .|> abs |> findmin
            val[d] = xGrids[d][res[2]]
            idx[d] = res[2]
        end
        valNew[i] = val
        idxNew[i] = idx
    end
    return (valNew, idxNew)
end




# ------------------------------------------------------------------------------
"""
    isstationary(A::AbstractMatrix)::Bool

Check if the VAR(1) coefficient matrix `A` is for a stationary process.
Returns `true` if the eigenvalues of `A` are all inside the unit circle.
Checked by the `max(abs(eigen(A).values)) < 1` condition.
"""
function isstationary(A::AbstractMatrix)::Bool
    return (eigen(A).values |> maximum |> abs) < 1
end










#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Discretization helpers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
    tauchen(
        N::Int, 
        ρ::Real, 
        σ::Real; 
        yMean::Real = 0.0, 
    	nσ   ::Real = 3,
    )::@NamedTuple{states::Vector{Float64}, probs::Matrix{Float64}}

Discretize an AR(1) process with mean `xMean`, persistence `ρ`, and standard
deviation `σ` into `N` states. The function returns a Markov chain with the
states and the transition probabilities.

`y_t = (1-ρ)*xMean + ρ*y_{t-1} + σ*ϵ_t, ϵ_t ~ N(0,1)`

The argument `nσ` is the number of standard deviations to include in the
discretization. The chosen states include the endpoints.

This method is based on the Tauchen (1986). It is the baseline method.

Tauchen, George. "Finite state markov-chain approximations to univariate and 
vector autoregressions." Economics letters 20, no. 2 (1986): 177-181.

This implementation is a modified version of:
`https://github.com/hendri54/shared/blob/master/%2Bar1LH/tauchen.m`
"""
function tauchen(
    N     ::Int,
    ρ     ::Real,
    σ     ::Real;
    yMean ::Real = 0.0,
    nσ    ::Real = 3,
)::@NamedTuple{states::Vector{Float64}, probs::Matrix{Float64}}
    @assert N > 1 "N must be > 1"
    @assert σ > 0 "σ must be > 0"
    @assert nσ > 0 "nσ must be > 0"
    @assert -1 < ρ < 1 "ρ must be in (-1,1) for staionary process"

    # Width of grid
    a_bar = nσ * sqrt(σ^2.0 / (1.0 - ρ^2))

    # Grid
    y = LinRange(-a_bar, a_bar, N)

    # Distance between points
    d = y[2] - y[1]

    # get transition probabilities
    trProbM = zeros(N, N)
    for iRow in 1:N
        # do end points first
        trProbM[iRow,1] = sf.normcdf((y[1] - ρ*y[iRow] + d/2) / σ)
        trProbM[iRow,N] = 1 - sf.normcdf((y[N] - ρ*y[iRow] - d/2) / σ)

        # fill the middle columns
        for iCol = 2:N-1

            trProbM[iRow,iCol] = (
                sf.normcdf((y[iCol] - ρ*y[iRow] + d/2) / σ) -
                sf.normcdf((y[iCol] - ρ*y[iRow] - d/2) / σ)
            )

        end # iCol
    end # iRow

    # normalize the probs to rowsum = 1 due to possible float errors
    trProbM ./= sum(trProbM, dims=2)

    # don't forget to shift the process to the position of the long-term mean
    return (
        states = y .+ yMean,
        probs  = trProbM
    )
end # tauchen

# TODO: Rouwenhorst (1995) discretization method, add later



























