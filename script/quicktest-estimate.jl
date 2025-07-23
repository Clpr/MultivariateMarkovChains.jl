# import MultivariateMarkovChains as mmc
using Statistics
mmc = include("../src/MultivariateMarkovChains.jl")


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OLS estimation of AR(1) processes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Frequency counting estimation of Multivariate Markov Chains
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Example data: already gridized
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



# Example data: not gridized
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
















