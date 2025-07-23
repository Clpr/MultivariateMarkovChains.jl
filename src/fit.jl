#===============================================================================
ESTIMATION METHODS

1. for AR(1)
    - OLS estimation
2. for VAR(1)
    - OLS estimation
3. for multivariate Markov chains
    - frequency counts
    - Young (2010)'s deterministic simulation method
===============================================================================#
export fit, fit!







# ==============================================================================
# ESTIMTION PROCEDURE: AR(1) PROCESS
# ==============================================================================
"""
    fit!(
        ar1  ::AR1, 
        xThis::AbstractVector, 
        xNext::AbstractVector ;
        x0   ::Real = 0.0
    )

Fit the AR(1) process `ar1` to the time series data `xThis` and `xNext`. The
function updates the parameters of the AR(1) process in-place. The time series
data `xThis` and `xNext` are arranged as: `xThis[t]` is `x(t)` and `xNext[t]`
is `x(t+1)`. Be careful with the time ordering. The length of `xThis` and
`xNext` must be the same.

The argument `x0` is the initial value of the process, which defaults to 0.0.
"""
function fit!(
    ar1  ::AR1, 
    xThis::AbstractVector, 
    xNext::AbstractVector ;
    x0   ::Real = 0.0
)
    N = length(xThis)
    @assert N == length(xNext) "xThis and xNext must have the same length"

    ρ, cst  = mvs.llsq(xThis, xNext, bias = true)
    xavg    = cst / (1 - ρ)
    xfitted = ρ .* xThis .+ (1 - ρ) * xavg
    resids  = xNext - xfitted
    σ       = Statistics.var(resids) |> sqrt

    ar1.x0   = x0
    ar1.ρ    = ρ
    ar1.xavg = xavg
    ar1.σ    = σ

    return nothing
end
# ------------------------------------------------------------------------------
"""
    fit!(ar1::AR1, xPath::AbstractVector ; x0::Real = 0.0)

Fit the AR(1) process `ar1` to the time series data `xPath`. The function update
the parameters of the AR(1) process in-place. The time series data `xPath` is
arranged as: `xPath[t]` is `x(t)`. Be careful with the time ordering.
"""
function fit!(ar1::AR1, xPath::AbstractVector ; x0::Real = 0.0)
    fit!(ar1, xPath[1:end-1], xPath[2:end], x0 = x0)
    return nothing
end # fit!
# ------------------------------------------------------------------------------
"""
    fit(xThis::AbstractVector, xNext::AbstractVector ; x0::Real = 0.0)

Fit an AR(1) process to the time series data `xThis` and `xNext`. The function
returns a new AR(1) process with the estimated parameters.
"""
function fit(xThis::AbstractVector, xNext::AbstractVector ; x0::Real = 0.0)
    ar1 = AR1()
    fit!(ar1, xThis, xNext, x0 = x0)
    return ar1
end # fit
# ------------------------------------------------------------------------------
"""
    fit(xPath::AbstractVector ; x0::Real = 0.0)

Fit an AR(1) process to the time series data `xPath`. The function returns a new
AR(1) process with the estimated parameters.
"""
function fit(xPath::AbstractVector ; x0::Real = 0.0)
    ar1 = AR1()
    fit!(ar1, xPath, x0 = x0)
    return ar1
end # fit









# ==============================================================================
# ESTIMTION PROCEDURE: VAR(1) PROCESS
# ==============================================================================
"""
    fit!(
        var1 ::VAR1{D},
        xThis::AbstractMatrix,
        xNext::AbstractMatrix ;
        x0   ::AbstractVector = zeros(D),
        dims ::Int = 1,
    ) where {D}

Fit the VAR(1) process `var1` to the time series data `xThis` and `xNext`. The
function updates the parameters of the VAR(1) process in-place. The time series
data `xThis` and `xNext` are arranged as: `xThis[:, t]` is `x(t)` and
`xNext[:, t]` is `x(t+1)`. Be careful with the time ordering and the dimension
arrangement. The number of columns of `xThis` and `xNext` must be the same.

The `dims` argument specifies how the data is arranged. If `dims = 1`, then
each row of `xThis` and `xNext` corresponds to a period's observation of size D;
if `dims = 2`, then each column of `xThis` and `xNext` corresponds to a period's
observation of size D. The default is `dims = 1`.
"""
function fit!(
    var1 ::VAR1{D},
    xThis::AbstractMatrix,
    xNext::AbstractMatrix ;
    x0   ::AbstractVector = zeros(D),
    dims ::Int = 1,
) where {D}
    @assert length(x0) == D "x0 must have $D elements"
    
    ρ_and_cst = mvs.llsq(
        dims == 1 ? xThis : xThis', 
        dims == 1 ? xNext : xNext', 
        bias = true, 
        dims = 1
    )
    ρ   = ρ_and_cst[1:D, 1:D]
    cst = ρ_and_cst[D+1, 1:D]

    xavg = (I(D) - ρ) \ cst

    resids = if dims == 1
        xfitted = ρ * xThis' .+ cst
        xNext' - xfitted
    elseif dims == 2
        xfitted = ρ * xThis .+ cst
        xNext - xfitted
    end
    Σ = Statistics.cov(resids, dims = dims) |> dst.Symmetric

    var1.x0   = x0
    var1.ρ    = ρ
    var1.xavg = xavg
    var1.Σ    = Σ

    return nothing
end
# ------------------------------------------------------------------------------
"""
    fit!(
        var1 ::VAR1{D},
        xPath::AbstractMatrix ;
        x0   ::AbstractVector = zeros(D),
        dims ::Int = 1,
    ) where {D}

Fit the VAR(1) process `var1` to the time series data `xPath`. The function
updates the parameters of the VAR(1) process in-place.
"""
function fit!(
    var1 ::VAR1{D},
    xPath::AbstractMatrix ;
    x0   ::AbstractVector = zeros(D),
    dims ::Int = 1,
) where {D}
    if dims == 1
        fit!(var1, xPath[1:end-1,:], xPath[2:end,:], x0 = x0, dims = dims)
    elseif dims == 2
        fit!(var1, xPath[:,1:end-1], xPath[:,2:end], x0 = x0, dims = dims)
    else
        error("dims must be either 1 or 2")
    end
    return nothing
end # fit!
# ------------------------------------------------------------------------------
"""
    fit(
        D::Int,
        xThis::AbstractMatrix,
        xNext::AbstractMatrix ;
        x0   ::AbstractVector = zeros(D),
        dims ::Int = 1,
    ) where {D}

Fit a VAR(1) process to the time series data `xThis` and `xNext`. The function
returns a new VAR(1) process with the estimated parameters.
"""
function fit(
    D::Int,
    xThis::AbstractMatrix,
    xNext::AbstractMatrix ;
    x0   ::AbstractVector = zeros(D),
    dims ::Int = 1,
)
    var1 = VAR1{D}()
    fit!(var1, xThis, xNext, x0 = x0, dims = dims)
    return var1
end
# ------------------------------------------------------------------------------
"""
    fit(
        D::Int,
        xPath::AbstractMatrix ;
        x0   ::AbstractVector = zeros(D),
        dims ::Int = 1,
    ) where {D}

Fit a VAR(1) process to the time series data `xPath`. The function returns a new
VAR(1) process with the estimated parameters.
"""
function fit(
    D::Int,
    xPath::AbstractMatrix ;
    x0   ::AbstractVector = zeros(D),
    dims ::Int = 1,
)
    var1 = VAR1{D}()
    fit!(var1, xPath, x0 = x0, dims = dims)
    return var1
end # fit

