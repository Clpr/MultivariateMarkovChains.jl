#===============================================================================
AR(1)/VAR(1) PROCESS REPRESENTATIONS

1. AR(1)
2. VAR(1) - covariance allowed

## interface
- Base.rand() - draw a sample path of the process
- merge()    - to glue two AR/VAR(1) processes to a higher-dim VAR(1) process
===============================================================================#
export AR1, VAR1



# ==============================================================================
# AR(1) PROCESS (SCALAR)
# ==============================================================================
"""
    AR1

A structure to represent an AR(1) process:

x(t+1) = ρ * x(t) + (1 - ρ) * xavg + σ * ε(t+1)

where `x` is a scalar time series. For vector-valued processes, use 
`VAR1`.

## Fields
- `x0`: Initial value of the process (default: 0.0)
- `ρ`: AR(1) coefficient (default: 1.0)
- `xavg`: Long-term average value (default: 0.0)
- `σ`: Standard deviation of the innovation shock (default: 1.0)

## Notes
- The default values correspond to a random walk.
- For log-normal processes, define the process on the log scale.

## Example
```julia
using Statistics
import MultivariateMarkovChains as mmc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AR(1) process representation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# INITIALIZATION ---------------------------------------------------------------

# define a standard random walk
ar1 = mmc.AR1()

# define an AR(1) process with exact parameters
ar1 = mmc.AR1(x0 = 0.0, ρ = 0.9, xavg = 0.0, σ = 0.1)


# STATISTICS -------------------------------------------------------------------

# unconditional mean
mean(ar1)

# unconditional variance & standard deviation
var(ar1)
std(ar1)

# if the process is stationary
mmc.isstationary(ar1)
mmc.isstationary(mmc.AR1(ρ = 1.1))


# SIMULATION -------------------------------------------------------------------

# simulate a sample path, overloads `Base.rand`
shocks = randn(100)

# scenario: given realized shocks
simulated_path = rand(ar1, shocks) # use default x0
simulated_path = rand(ar1, shocks, x0 = 1.0) # specify initial value

# scenario: generic simulation
simulated_path = rand(ar1, 100) # simulate 100 steps with default x0
simulated_path = rand(ar1, 100, x0 = 1.0)
simulated_path = rand(ar1, 100, seed = 123) # specify random seed

```
"""
Base.@kwdef mutable struct AR1
    x0  ::Float64 = 0.0          # Initial value
    ρ   ::Float64 = 1.0          # AR(1) coefficient
    xavg::Float64 = 0.0          # Long-term average value
    σ   ::Float64 = 1.0          # Standard deviation of the innovation shock
end # AR1
# ------------------------------------------------------------------------------
function Base.show(io::IO, ar1::AR1)
    println(io, "AR(1) Process:")
    println(io, "  Initial value          : $(ar1.x0)")
    println(io, "  AR(1) coefficient      : $(ar1.ρ)")
    println(io, "  Long-term average value: $(ar1.xavg)")
    println(io, "  Volatility             : $(ar1.σ)")
    println(io, "  Stationary?            : ", ar1.ρ < 1.0)
    println(io, "  Process equation:")
    @printf(io, "  x(t+1) = %+.2f * x(t) + (1 %+.2f) * %+.2f %+.2f * ε(t+1)\n", 
        ar1.ρ, -ar1.ρ, ar1.xavg, ar1.σ
    )
end # show
# ------------------------------------------------------------------------------
function Base.ndims(ar1::AR1)
    return 1
end # ndims
# ------------------------------------------------------------------------------
"""
    rand(
        ar1   ::AR1, 
        shocks::AbstractVector ; 
        x0    ::Float64 = ar1.x0,
    )::Vector{Float64}

Simulate an AR(1) process given a vector of shocks. The shocks vector must have
at least one element. The first value of the output vector is always `x0`, and
subsequent values are generated based on the AR(1) process equation using the
shocks provided. The shocks vector is assumed to be of length `T-1`.

Unless necessary, the `shocks` vector should be drawn from N(0,1) distribution.
"""
function Base.rand(
    ar1   ::AR1, 
    shocks::AbstractVector ; 
    x0    ::Float64 = ar1.x0,
)::Vector{Float64}
    T = length(shocks) + 1
    @assert T > 0 "Shocks vector must have at least one element"

    if T == 1; return Float64[x0]; end
    xpath    = Vector{Float64}(undef, T)
    xpath[1] = x0
    for t in 2:T
        meanpart = ar1.ρ * xpath[t-1] + (1 - ar1.ρ) * ar1.xavg
        xpath[t] = meanpart + ar1.σ * shocks[t-1]
    end
    return xpath
end # rand
# ------------------------------------------------------------------------------
"""
    rand(ar1::AR1, T::Int; ...)

Simulates an AR(1) process for `T` time steps, starting from the initial value 
`x0`. Returns a vector of length `T` containing the simulated values. The first
value is always `x0`, and subsequent values are generated based on the AR(1)
process equation.
"""
function Base.rand(
    ar1 ::AR1, 
    T   ::Int ; 
    x0  ::Float64 = ar1.x0,
    seed::Union{Nothing,Int} = nothing,
)::Vector{Float64}
    @assert T > 0 "T must be a positive integer"
    shocks   = randn(MersenneTwister(seed), T-1)
    return rand(ar1, shocks, x0 = x0)
end # rand
# ------------------------------------------------------------------------------
"""
    mean(ar1::AR1)::Float64

Computes the unconditional mean of the AR(1) process.
"""
function Statistics.mean(ar1::AR1)::Float64
    return ar1.xavg
end # mean
# ------------------------------------------------------------------------------
"""
    var(ar1::AR1)::Float64

Computes the population variance of the AR(1) process say `var(x(t))`.
"""
function Statistics.var(ar1::AR1)::Float64
    return ar1.σ^2 / (1 - ar1.ρ^2)
end # var
# ------------------------------------------------------------------------------
"""
    std(ar1::AR1)::Float64

Computes the population standard deviation of the AR(1) process say `std(x(t))`.
"""
function Statistics.std(ar1::AR1)::Float64
    return sqrt(Statistics.var(ar1))
end # std
# ------------------------------------------------------------------------------
"""
    isstationary(ar1::AR1)::Bool

Check if the AR(1) process is stationary. Returns `true` if the absolute value
of the AR(1) coefficient `ρ` is less than 1, locating in the unit circle.
"""
function isstationary(ar1::AR1)::Bool
    return abs(ar1.ρ) < 1.0
end # isstationary












# ==============================================================================
# VAR(1) PROCESS (VECTOR-VALUED)
# ==============================================================================
"""
    VAR1{D}

A structure to represent a VAR(1) process in `D` dimensions:

x(t+1) = ρ * x(t) + (1 - ρ) * xavg + E(t+1), E(t+1) ~ MvN(0, Σ)

where `x` is a vector-valued time series. `ρ` is a square matrix of size `D x D`
which is the AR(1) coefficient matrix, `xavg` is a vector of size `D`, and
`σ` is a covariance matrix of size `D x D`. And `E` is a vector of normal shocks
drawn from N(0,Σ) where Σ is the covariance matrix (not volatility matrix).

For scalar processes, use `AR1`.

The default values correspond to a random walk in `D` dimensions:

x(t+1) = I * x(t) + (1 - I) * 0 + Z(t+1), Z(t+1) ~ MvN(0, I)

where `I` is the identity matrix of size `D x D`.


## Example
```julia
using Statistics
import MultivariateMarkovChains as mmc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VAR(1) process representation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# INITIALIZATION ---------------------------------------------------------------

# define a standard 4-dimensional random walk
ar1 = mmc.VAR1{4}()

# define an AR(1) process with exact parameters
# accepts various types of inputs, e.g.
ar1 = mmc.VAR1{4}(
    x0   = LinRange(1,4,4),
    ρ    = fill(0.9,4) |> mmc.diagm, 
    xavg = zeros(4), 
    Σ    = [
        1.0 0.1 0.1 0.1;
        0.1 1.0 0.1 0.1;
        0.1 0.1 1.0 0.1;
        0.1 0.1 0.1 1.0;
    ] # covariance matrix, not volatility matrix
)

# META INFORMATION -------------------------------------------------------------

# dimensionality
ndims(ar1) # 4 in this example


# STATISTICS -------------------------------------------------------------------

# unconditional mean
mean(ar1)

# unconditional covariance matrix & variance vector
cov(ar1)
var(ar1)


# if the process is stationary
mmc.isstationary(ar1)



# SIMULATION -------------------------------------------------------------------

# scenario: generic simulation
simulated_path = rand(ar1, 100) # simulate 100 steps with default x0
simulated_path = rand(ar1, 100, x0 = rand(4))
simulated_path = rand(ar1, 100, seed = 123) # specify random seed


# scenario: given realized shocks (Distributions.jl is imported as `dst`)
shocks = rand(mmc.dst.MvNormal(ar1.Σ), 100) # draw from the correct distribution

simulated_path = rand(ar1, shocks)
simulated_path = rand(ar1, shocks, x0 = rand(4)) # specify initial value


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merging operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NOTE: the merging operation always assume marginal independence of the 
# processes being merged (while the origional covariance structure of the merged
# ingridient VAR(1) processes is preserved). If you want to ADD new covariance
# structure, consider directly constructing a VAR(1) process with the desired
# covariance matrix.

proc1 = mmc.AR1(ρ = 0.9, xavg = 0.0, σ = 0.1)
proc2 = mmc.AR1(ρ = 0.8, xavg = 1.0, σ = 0.2)
proc3 = mmc.VAR1{3}(
    x0 = [1.0, 2.0, 3.0],
    ρ = [0.9 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.7],
    xavg = [0.5, 1.5, 2.5],
    Σ = [
        1.0 0.1 0.1;
        0.1 1.0 0.2;
        0.1 0.2 1.0;
    ]
)

# merge two INDEPENDENT AR(1) processes into a VAR(1) process of dimension 2
merged_proc = merge(proc1, proc2)

# merge independent AR1 & a VAR1{3} into a VAR(1) process of dimension 1+3=4
# NOTE: the order of merging matters which determines the order of the states
merged_proc = merge(proc1, proc3)
merged_proc = merge(proc3, proc1)

# merge two independent VAR(1) processes into another VAR(1) process
merged_proc = merge(proc3, proc3)

```
"""
mutable struct VAR1{D}
    x0  ::SizedVector{D,Float64}  
    ρ   ::SizedMatrix{D,D,Float64}
    xavg::SizedVector{D,Float64}  

    # covariance matrix
    Σ   ::SizedMatrix{D,D,Float64}

    function VAR1{D}(;
        x0  ::AbstractVector = D |> zeros,
        ρ   ::AbstractMatrix = I(D),
        xavg::AbstractVector = D |> zeros,
        Σ   ::AbstractMatrix = I(D),
    ) where {D}
        return new{D}(
            x0   |> SizedVector{D,Float64},
            ρ    |> SizedMatrix{D,D,Float64},
            xavg |> SizedVector{D,Float64},
            Σ    |> SizedMatrix{D,D,Float64},
        )
    end
end # VAR1
# ------------------------------------------------------------------------------
function Base.show(io::IO, var1::VAR1{D}) where {D}
    println(io, "VAR(1) Process of dimension $D:")
    println(io, " Initial value          : $(var1.x0)")
    println(io, " AR(1) coefficient      : $(var1.ρ)")
    println(io, " Long-term average value: $(var1.xavg)")
    println(io, " Covariance matrix      : $(var1.Σ)")
    println(io, " Stationary?            : ", isstationary(var1.ρ))
    println(io, " Process equation:")
    println(io, " x(t+1) = ρ * x(t) + (1-ρ) * xavg + E(t+1), E(t+1) ~ MvN(0,Σ)")
end # show
# ------------------------------------------------------------------------------
function Base.ndims(var1::VAR1{D}) where {D}
    return D
end # ndims
# ------------------------------------------------------------------------------
function Statistics.mean(var1::VAR1{D}) where {D}
    return var1.xavg
end # mean
# ------------------------------------------------------------------------------
function Statistics.cov(var1::VAR1{D}) where {D}
    # NOTES: by definition, the covariance matrix Γ of X(t) satisfies
    # the following Lyapunov equation:
    # Γ = ρ * Γ * ρ' + Σ
    # analytical solution: vec(Γ) = (I - kron(ρ,ρ))^(-1) * vec(Σ)

    return reshape(
        (I - kron(var1.ρ, var1.ρ)) \ vec(var1.Σ),
        D, D
    )
end # cov
# ------------------------------------------------------------------------------
function Statistics.var(var1::VAR1{D}) where {D}
    return Statistics.cov(var1) |> diag
end # var
# ------------------------------------------------------------------------------
"""
    isstationary(var1::VAR1{D}) where {D}

Check if the VAR(1) process is stationary. Returns `true` if the maximum
absolute eigenvalue of the AR(1) coefficient matrix `ρ` is less than 1.
"""
function isstationary(var1::VAR1{D}) where {D}
    return isstationary(var1.ρ)
end # isstationary
# ------------------------------------------------------------------------------
"""
    rand(
        ar1   ::VAR1{D}, 
        shocks::AbstractMatrix ; 
        x0    ::AbstractVector = ar1.x0,
    )::Matrix{Float64} where {D}

Simulate an VAR(1) process given a vector of shocks. The shocks vector must have
at least one element. The first value of the output vector is always `x0`, and
subsequent values are generated based on the AR(1) process equation using the
shocks provided. The shocks matrix is assumed to be of size `D x (T-1)`.

Returns a matrix of size `D x T`, where `T` is the number of periods.

Unless necessary, the `shocks` matrix should be drawn from MvN(0,Σ) distribution
to correctly reflect the covariance structure of the process.
"""
function Base.rand(
    ar1   ::VAR1{D}, 
    shocks::AbstractMatrix ; 
    x0    ::AbstractVector = ar1.x0,
)::Matrix{Float64} where {D}
    T = size(shocks,2) + 1
    @assert T > 0 "Shocks vector must have at least one element"

    if T == 1; return reshape(x0,D,1); end
    xpath = Matrix{Float64}(undef, D, T)
    xpath[:,1] = x0
    for t in 2:T
        meanpart = ar1.ρ * xpath[:,t-1] + (I(D) - ar1.ρ) * ar1.xavg
        xpath[:,t] = meanpart + shocks[:,t-1]
    end
    return xpath
end # rand
# ------------------------------------------------------------------------------
"""
    rand(ar1::AR1, T::Int; ...)

Simulates an AR(1) process for `T` time steps, starting from the initial value 
`x0`. Returns a vector of length `T` containing the simulated values. The first
value is always `x0`, and subsequent values are generated based on the AR(1)
process equation.
"""
function Base.rand(
    ar1 ::VAR1{D}, 
    T   ::Int ; 
    x0  ::AbstractVector = ar1.x0,
    seed::Union{Nothing,Int} = nothing,
)::Matrix{Float64} where {D}
    @assert T > 0 "T must be a positive integer"
    shocks = rand(
        MersenneTwister(seed), 
        dst.MvNormal(ar1.Σ),
        T-1
    )
    return rand(ar1, shocks, x0 = x0)
end # rand











# ==============================================================================
# MERGE
# NOTE: CAN ONLY MERGE INDEPENDENT PROCESSES
# ==============================================================================
"""
    merge(ar1::AR1, ar2::AR1)::VAR1{2}

Merge two independent AR(1) processes into a VAR(1) process of dimension 2.

If covariance structure is needed, consider directly constructing a `VAR1{2}`
while manually specifying the covariance matrix.
"""
function Base.merge(ar1::AR1, ar2::AR1)::VAR1{2}
    return VAR1{2}(
        x0 = [ar1.x0, ar2.x0],
        ρ  = [
            ar1.ρ 0.0;
            0.0 ar2.ρ;
        ],
        xavg = [ar1.xavg, ar2.xavg],
        Σ = [
            ar1.σ^2 0.0;
            0.0 ar2.σ^2;
        ]
    )
end # merge
# ------------------------------------------------------------------------------
"""
    merge(ar1::AR1, var1::VAR1{D})::VAR1{D+1} where {D}
    merge(var1::VAR1{D}, ar1::AR1)::VAR1{D+1} where {D}

Merge an independent AR(1) process with a VAR(1) process of dimension `D`
into a VAR(1) process of dimension `D+1`. The covariance structure is preserved
from the VAR(1) process, and the AR(1) process is added as the first or the last
dimension, respectively.
"""
function Base.merge(ar1::AR1, var1::VAR1{D}) where {D}
    ρ = zeros(D+1,D+1)
    ρ[1,1] = ar1.ρ
    ρ[2:end, 2:end] = var1.ρ

    Σ = zeros(D+1,D+1)
    Σ[1,1] = ar1.σ^2
    Σ[2:end, 2:end] = var1.Σ

    return VAR1{D+1}(
        x0 = [ar1.x0; var1.x0],
        xavg = [ar1.xavg; var1.xavg],
        ρ = ρ, Σ = Σ
    )
end
function Base.merge(var1::VAR1{D}, ar1::AR1) where {D}
    ρ = zeros(D+1,D+1)
    ρ[1:D, 1:D] = var1.ρ
    ρ[end,end] = ar1.ρ

    Σ = zeros(D+1,D+1)
    Σ[1:D, 1:D] = var1.Σ
    Σ[end,end] = ar1.σ^2

    return VAR1{D+1}(
        x0 = [var1.x0; ar1.x0],
        xavg = [var1.xavg; ar1.xavg],
        ρ = ρ, Σ = Σ
    )
end
# ------------------------------------------------------------------------------
"""
    Base.merge(var1::VAR1{D1}, var2::VAR1{D2}) where {D1,D2}

Merge two independent VAR(1) processes into another VAR(1) process of dimension
`D1+D2`. The covariance structure is preserved from both processes while the two
processes are independent (consequently, the covariance matrix is blockdiagonal)

If covariance structure is needed, consider directly constructing an instance of
`VAR1{D1+D2}` with the desired covariance matrix.
"""
function Base.merge(var1::VAR1{D1}, var2::VAR1{D2}) where {D1,D2}
    ρ = zeros(D1+D2, D1+D2)
    ρ[1:D1, 1:D1] = var1.ρ
    ρ[D1+1:end, D1+1:end] = var2.ρ

    Σ = zeros(D1+D2, D1+D2)
    Σ[1:D1, 1:D1] = var1.Σ
    Σ[D1+1:end, D1+1:end] = var2.Σ

    return VAR1{D1+D2}(
        x0 = [var1.x0; var2.x0],
        xavg = [var1.xavg; var2.xavg],
        ρ = ρ, Σ = Σ
    )
end


