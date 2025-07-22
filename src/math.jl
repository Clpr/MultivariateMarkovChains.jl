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
    inv_discrete_distribution(pr::Real, probs::Vector{Float64})::Int

Inverse discrete distribution function. Given a probability `pr` and a
discrete distribution `probs`, this function returns the index of the first
cumulative probability that is greater than or equal to `pr`. The `probs`
must be a vector of probabilities that sum to 1.0, and `pr` must be in the
range [0, 1].
"""
function inv_discrete_distribution(pr::Real, probs::Vector{Float64})::Int
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
    @assert length(xGrid) > 0 "xGrid must have at least one element"
    @assert all(unique(xGrid) .== xGrid) "xGrid must be unique"

    return getindex.(argmin(abs.(Xs .- xGrid'), dims = 2), 2)
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
        trProbM[iRow,1] = normcdf((y[1] - ρ*y[iRow] + d/2) / σ)
        trProbM[iRow,N] = 1 - normcdf((y[N] - ρ*y[iRow] - d/2) / σ)

        # fill the middle columns
        for iCol = 2:N-1

            trProbM[iRow,iCol] = (
                normcdf((y[iCol] - ρ*y[iRow] + d/2) / σ) -
                normcdf((y[iCol] - ρ*y[iRow] - d/2) / σ)
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










#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Young (2010) deterministic simulation method
# 
# Young, Eric R. "Solving the incomplete markets model with aggregate 
# uncertainty using the Krusell–Smith algorithm and non-stochastic simulations."
# Journal of Economic Dynamics and Control 34, no. 1 (2010): 36-41.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






















