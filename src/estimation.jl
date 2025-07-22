#===============================================================================
ESTIMATION HELPERS

1. for AR(1)
2. for VAR(1)
3. for multivariate Markov chains
    - OLS
    - Young (2010)'s deterministic simulation method
===============================================================================#










# ------------------------------------------------------------------------------
"""
    AR1(xThis::AbstractVector, xNext::AbstractVector ; x0::Real = 0.0)

Defines an AR(1) process using two vectors: `xThis` and `xNext`, which represent
the current and next time steps, respectively. This function is useful for pair-
like data. No time ordering is assumed but the two vectors have implied this as
in their definition.

The argument `x0` is the initial value of the process, which defaults to 0.0.

## Example
```julia
xThis = [1.0, 2.0, 3.0]
xNext = [1.1, 2.1, 3.1]
ar1 = AR1(xThis, xNext)
```
"""
function AR1(
    xThis::AbstractVector, 
    xNext::AbstractVector ;
    x0   ::Real = 0.0,
)
    N = length(xThis)
    @assert N == length(xNext) "xThis and xNext must have the same length"

    ρ, cst  = mvs.llsq(xThis, xNext, bias = true)
    xavg    = cst / (1 - ρ)
    xfitted = ρ .* xThis .+ (1 - ρ) * xavg
    resids  = xNext - xfitted
    σ       = Statistics.var(resids) |> sqrt
    return AR1(x0, ρ, xavg, σ)
end # AR1
# ------------------------------------------------------------------------------
"""
    AR1(xpath::AbstractVector ; x0::Real = xpath[1])

Defines an AR(1) process by fitting the parameters to a given time series data.
The length of `xpath` must be at least 3 as the process requires at least three
degree of freedom for `(ρ,xavg,σ)`.

The xpath should be time-ascending, i.e., `xpath[1]` is the initial value at 
time `t=0`.

The argument `x0` is the initial value of the process, which defaults to the
first element of `xpath`.

## Example
```julia
ar1 = AR1(ρ = 0.9, σ = 0.1)
xpath = ch2.simulate(tmp, 100)
tmp = ch2.AR1(xpath)
```
"""
function AR1(xpath::AbstractVector ; x0::Real = xpath[1])
    return AR1(xpath[1:end-1], xpath[2:end], x0 = x0)
end # AR1