# import MultivariateMarkovChains as mmc
using Statistics
mmc = include("../src/MultivariateMarkovChains.jl")


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

















