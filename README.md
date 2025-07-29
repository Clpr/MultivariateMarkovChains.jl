# MultivariateMarkovChains



**MultivariateMarkovChains.jl** is a Julia package that provides tools for working with multivariate finite discrete state time-homogeneous Markov chains and other relative stochastic processes, including scalar AR(1) and vector VAR(1). The package offers functionality for definition, simulation, parameter estimation, and conversion of the above stochastic processes. It also supports merging independent processes into higher-dimensional systems, making it particularly useful for macroeconomic modeling, financial econometrics, and other applications requiring the analysis of multivariate time series with complex interdependencies.

To install this package do:

```julia
pkg> add MultivariateMarkovChains.jl
```

Or install it from GitHub directly:

```julia
pkg> add "https://github.com/Clpr/MultivariateMarkovChains.jl.git"
```

## Usage

This section introduces the several core data structures and the usage examples,
then it shows how to convert from one to another.


### Multivariate Markov chain

A time-homogeneous mutlivariate Markov chain describes how a vector random variable $x_t$
transits across time, particularly discrete time periods.

$$
x \in (x^1,x^2,\dots,x^N), x^i \in \mathbb{R}^D, \forall i
$$

Such a Markov chain can be denoted by a 2-tuple of a state vector $\mathbf{X} := \{x^1,x^2,\dots,x^N\}$ and a N-by-N transition matrix $\mathbf{P} \in\mathbb{R}^{N\times N}$, in which `P[i,j]` is the probability transiting from state $i$ to state $j$.

```julia
import Statistics
import MultivariateMarkovChains as mmc


# INITIALIZATION ---------------------------------------------------------------

# define a multivariate Markov chain with 2 states in 2D
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

# copy & deepcopy
copy(mc)
deepcopy(mc)

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



# ANALYSIS ---------------------------------------------------------------------

# solves the stationary distribution
ss = mmc.stationary(mc)

# computes the long-term mean of the states
Statistics.mean(mc)

```






### AR(1) process

$$
x_{t+1} = \rho \cdot x_{t} + (1-\rho) \cdot x_{avg} + \sigma \cdot \varepsilon_{t+1}, \varepsilon_{t+1} \sim N(0,1)
$$

where $\rho$ is the autoregression coefficient which implies a stationary process if $|\rho|<1$ locates in the unit circle; $x_{avg}$ is the unconditional long-term mean of any $x(t)$; $\sigma$ is the volatility of the error term (or say innovarion shock).

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

# MANUPULATION -----------------------------------------------------------------

copy(ar1)
deepcopy(ar1)

```

### VAR(1) process

$$
X_{t+1} = \rho \cdot X_{t} + (I - \rho) \cdot X_{avg} + E_{t+1}, E_{t+1} \sim MvNormal(0_{D}, \Sigma)
$$

where $X_{t+1} \in\mathbb{R}^D$ is a D-dimensional state vector;
$\rho \in \mathbb{R}^{D \times D}$ is the autoregression coefficient (matrix);
$I$ is the identity matrix; $X_{avg}  \in\mathbb{R}^D$ is the unconditional mean;
and $E_{t+1}$ is the innovarion shock that follows a D-dimensional zero-mean multivariate normal distribution of covariance matrix $\Sigma \in \mathbb{R}^{D,D}$.

This package allows to model the covariance structure among dimensions through a non-necessarily diagonal covariance matrix $\Sigma$.


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


# MANUPULATION -----------------------------------------------------------------

copy(ar1)
deepcopy(ar1)


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





### Converting one process to another

Beyond direct parameter estimation, `MultivariateMarkovChain` objects can be constructed from other processes using discretization algorithms:

**From AR(1) processes:**

- Tauchen (1986) method
- Rouwenhorst (1995) method *(not yet implemented)*

**From VAR(1) processes:**

- Modified Rouwenhorst method *(not yet implemented)*

**From continuous state mappings:**

- Young (2010) non-stochastic simulation
- Gridization + frequency count (check the estimation section for `fit!()`)


```julia
import MultivariateMarkovChains as mmc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tauchen (1986)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ar1 = mmc.AR1(ρ = 0.9, xavg = 10.0, σ = 0.1)

# style 1: OOP, returns a `MultivariateMarkovChain`
mmc.tauchen(
    ar1,
    5, # number of discretized states
    nσ = 2.5, # spanning how many standard deviation around the mean
)

# style 2: generic, returns a NamedTuple of `states` and `probs`
mmc.tauchen(
    5, # number of discretized states
    ar1.ρ, # persistence
    ar1.σ, # standard deviation
    yMean = ar1.xavg, # mean
    nσ = 2.5, # spanning how many standard deviation around the mean
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Young (2010) non-stochastic simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# an example 2-D operator mapping continuous vector states to the same space
# e.g. consider [0,1]^2 space for simplicity
# e.g. do flipping around (0.5,0.5) for example
f2fit(X) = clamp.([1.0 - X[1], 1.0 - X[2]], 0.0, 1.0)

# design grids for discretizing the continuous state space
grids = [
    LinRange(0,1,11), # 11 points in the first dimension
    LinRange(0,1,21), # 21 points in the second dimension
]

# style 1: let the package do the work
# returns a `MultivariateMarkovChain{2}` with 11*21 = 231 states
mc = mmc.young(
    f2fit, # the operator mapping
    grids, # the grids for discretization
)

# style 2: break down the steps and control more details
# returns a NamedTuple of `states` and `probs` which can be used to construct 
# a `MultivariateMarkovChain`

# step 2.1: evaluate the operator for every grid point, Cartesian/tensor joined
# and stack the results for convenience
Xthis = Iterators.product(grids...) |> collect .|> collect |> vec
Xnext = [f2fit(xy) for xy in Xthis] |> stack

# step 2.2: gridize/snap the next states to the grids; each dimension has its own grid
Xnext_gridized = [
    [x1,x2]
    for (x1,x2) in zip(
        grids[1][mmc.gridize(Xnext[1,:], grids[1])],
        grids[2][mmc.gridize(Xnext[2,:], grids[2])],
    )
]

# step 2.3: count the frequency of state transitions
# tips: use DataStructures.Counter
freqs = mmc.DataStructures.counter(Xthis .=> Xnext_gridized)

# step 2.4: construct the transition probability matrix
Pr = Float64[
    freqs[xi => xj]
    for (xi,xj) in Iterators.product(Xthis,Xthis)
]
# Pr |> mmc.sparse # (optional) check the sparsity pattern
Pr ./= sum(Pr, dims = 2) # normalize the rows to sum to 1


# step 2.5: construct the `MultivariateMarkovChain`
mc = mmc.MultivariateMarkovChain(
    Xthis, # the states
    Pr,    # the transition probability matrix
    validate  = true, # validate the transition probabilities
    normalize = true, # normalize the rows to sum to 1
)

# Then, if the mapping `f2fit` is stochastic, one should merge the constructed
# `MultivariateMarkovChain` with the Markov chain of the stochastic states.


```













### Estimating the processes from observational data

While this package primarily focuses on data structures and model representations, it includes basic parameter estimation methods for fitting models to observed data. **Note**: These estimation methods are intended for prototyping and quick testing purposes only. Bugs may happen. For production use or rigorous statistical analysis, consider using specialized econometric packages.


```julia
import MultivariateMarkovChains as mmc
using Statistics


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Frequency counting estimation of Multivariate Markov Chains
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Example data: already gridized -----------------------------------------------
xPath = stack(
    rand(
        [[1,1], [1,2], [2,1], [2,2]], 
        100
    ),
    dims = 1
)
xThis = xPath[1:end-1,:]
xNext = xPath[2:end,:]


mc = mmc.MultivariateMarkovChain(
    [[1,1],[1,2],[1,3]],  # just init something for updating, but the dimensionality should match
    rand(3,3),
    normalize = true,
)


mmc.fit!(mc, xThis, xNext, dims = 1) # `dims = 1` indicates that each row is an observation
mmc.fit!(mc, xThis', xNext', dims = 2)
mmc.fit!(mc, xPath, dims = 1) # the whole time series, the order matters

sum(mc.Pr, dims = 2) |> display # check the row sums


mc1 = mmc.fit(xThis, xNext, dims = 1) # fit a new Markov chain
mc2 = mmc.fit(xPath, dims = 1) # fit a new Markov chain



# Example data: not gridized ---------------------------------------------------
xPathContinuous = rand(100, 2)

# you may use `mmc.gridize` to gridize the data with a discrete grid
# out-of-grid data will be mapped to the nearest grid point

# usage 1 (gridize a vector)
mmc.gridize(rand(20), 0.0:0.1:1.0)

# usage 2 (gridize each row/column of a matrix)
# specify the grid for each row/column
# one can `stack` the returned vectors to a gridized matrix
mmc.gridize.(
    rand(100000,10) |> eachcol,
    [
        0.0:0.1:1.0
        for _ in 1:10
    ]
)

# now, let's gridize the non-gridized data
xPath = mmc.gridize.(
    xPathContinuous |> eachcol,
    [
        [0.0, 0.5, 1.0], # grid for the 1st dimension
        0.0:0.2:1.0    ,  # grid for the 2nd dimension
    ]
) |> stack

# then, we can fit a multivariate Markov chain
mc = mmc.fit(xPath, dims = 1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OLS estimation of AR(1) processes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc1 = mmc.AR1(ρ = 0.9, xavg = 0.0, σ = 0.1)
proc2 = mmc.AR1(ρ = 0.8, xavg = 1.0, σ = 0.2)
proc3 = mmc.VAR1{3}(
    x0 = [1.0, 2.0, 3.0],
    ρ  = [0.9 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.7],
    xavg = [0.5, 1.5, 2.5],
    Σ = [
        1.0 0.1 0.1;
        0.1 1.0 0.2;
        0.1 0.2 1.0;
    ]
)


# sample an AR(1) data path
xData = let T = 200
    ρ  = 0.9
    σ  = 0.233
    x0 = 1.0
    xavg = 1.0

    path = zeros(T)
    path[1] = x0
    for t in 2:T
        path[t] = ρ * path[t-1] + (1 - ρ) * xavg + σ * randn()
    end

    path
end


# fit & new
ar1 = mmc.fit(xData, x0 = 1.0) # a whole time series
ar1 = mmc.fit(xData[1:end-1], xData[2:end], x0 = 1.0) # today => tomorrow pairs


# in-place fit/update
mmc.fit!(proc1, xData, x0 = 1.0) # a whole time series
mmc.fit!(proc2, xData[1:end-1], xData[2:end], x0 = 1.0) # today (X) => tomorrow (Y) pairs



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OLS estimation of VAR(1) processes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

proc4 = mmc.VAR1{3}()

# sample an VAR(1) data path
xData = rand(proc3, 200)

# fit & new
proc5 = mmc.fit(
    3, # specify the dimensionality
    xData[:,1:end-1], # X(t) observations
    xData[:,2:end],   # X(t+1) observations
    x0 = zeros(3),
    dims = 2, # indicates that each column is an observation
)
proc5 = mmc.fit(
    3,
    xData,   # the whole time series
    x0 = zeros(3),
    dims = 2,
)


# in-place fit/update
mmc.fit!(
    proc4,
    xData[:,1:end-1], # X(t) observations
    xData[:,2:end],   # X(t+1) observations
    x0 = zeros(3),
    dims = 2, # indicates that each column is an observation
)
mmc.fit!(
    proc4,
    xData,   # the whole time series
    x0 = zeros(3),
    dims = 2,
)

```


















## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Reference

- Tauchen, George. “Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions.” Economics Letters 20, no. 2 (1986): 177–81. https://doi.org/10.1016/0165-1765(86)90168-0.
- Rouwenhorst, K. Geert. "Asset pricing implications of equilibrium business cycle models." Frontiers of business cycle research 1 (1995): 294-330.
- Young, Eric R. “Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell–Smith Algorithm and Non-Stochastic Simulations.” Journal of Economic Dynamics and Control 34, no. 1 (2010): 36–41. https://doi.org/10.1016/j.jedc.2008.11.010.
- Gospodinov, Nikolay, and Damba Lkhagvasuren. “A MOMENT‐MATCHING METHOD FOR APPROXIMATING VECTOR AUTOREGRESSIVE PROCESSES BY FINITE‐STATE MARKOV CHAINS.” Journal of Applied Econometrics 29, no. 5 (2014): 843–59. https://doi.org/10.1002/jae.2354.
- Kopecky, Karen A., and Richard MH Suen. "Finite state Markov-chain approximations to highly persistent processes." Review of Economic Dynamics 13, no. 3 (2010): 701-714.


