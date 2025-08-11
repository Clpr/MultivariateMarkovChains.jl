mmc = include("../src/MultivariateMarkovChains.jl")
# import MultivariateMarkovChains as mmc


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




#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Young (2010) non-stochastic simulation

--------------------
## Scenario: 

Deterministic continuous mapping that decides how a state X transits to X'

where:

X' = f(X) --> endogenous states 

and f is the (decision) rules that affects the transition matrix.

--------------------
## Notes:

- X can be vector-valued.
- No stochasticity. `f` defines a deterministic system.

--------------------
## In economics:

- Many economies without uncertainty fits into this idea.
- Compared with gridize the function `f`, approximating it with a Markov chain
  provides some degrees of boudned rationality, which might be interesting in
  studying precautionary behaviors.

--------------------
## Preparation for calling the API:

- A decision rule `X' = f(X)` that deicdes how the endogenous states X change 
  across any two periods. The prime mark denotes "next period". Practically, you
  can wrap an interpolant or approximator with a function and pass it to the API
- A collection of grids for X, dimension by dimension.

--------------------
## Illustrative example below:

- X: 2-vector in [0,1]^2
    - rule: Xp equals to the average of X, Q and Z (to introduce the joint depe-
      ndence on X, Q, and Z with the least effort)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

# prepare grid spaces
xgrids = [
    LinRange(0,1, 10),
    LinRange(0,1, 10),
]


# prepare the mapping X' = f(X) which returns continous values but
# not necessary to locate on the grid points
ftest(X) = begin
    
    # step: compute X' = f(X,Z)
    Xp = sqrt.(X)

    return Xp
end

# run the algorithm; parallization is available
@time mcY = mmc.young1(ftest, xgrids)





#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Young (2010) non-stochastic simulation

--------------------
## Scenario: 

Controlled Markov chain Y := (X,Z)

where:

X' = f(X,Z)               --> endogenous states 
Z  ~ exogenous Markov chain --> exogenous shock

and f is the (decision) rules that affects the transition matrix.

--------------------
## Notes:

- Any of X,Z can be vector-valued.
- `Z` is the exogenous stochasticity that drives the randomness of the process.

--------------------
## In economics:

- Many general equilibrium models that has similar structure such as the famous 
  Krusell-Smith model. Especially, all endogenous states can be explicitly upda-
  ted.
- In such economies:
    - X: typical aggregate state variables such as capital, asset holding
    - Z: typical aggregate uncertainties such as TFP shock
- The multivariate Markov chain Y = (X,Z) gives a full picture about how the
  aggregate dynamics evolves.

--------------------
## Preparation for calling the API:

- A decision rule `X' = f(X,Z)` that deicdes how the endogenous states X change 
  across any two periods. The prime mark denotes "next period". Practically, you
  can wrap an interpolant or approximator with a function and pass it to the API
- A collection of grids for X, dimension by dimension.
- A MultivariateMarkovChain that describes the transition of Z shocks.

--------------------
## Illustrative example below:

- X: 2-vector in [0,1]^2
    - rule: Xp equals to the average of X, Q and Z (to introduce the joint depe-
      ndence on X, Q, and Z with the least effort)
- Z: 2-vector in [0,1]^2 
    - rule: follows a VAR(1); WLOG, assume cov(Z) = 0 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

# prepare grid spaces
xgrids = [
    LinRange(0,1, 10),
    LinRange(0,1, 10),
]

# prepare the Z's process
Zproc = let

    # fake a AR(1) transition matrix for one dimension of Z
    pr = mmc.tauchen(
        3, # no. point per dimension of Z
        0.9, # AR(1) coefficient
        0.1, # Std Var of the error term
    ).probs
    zs = [[z,] for z in LinRange(0,1, pr |> size |> first)]

    mcZi = mmc.MultivariateMarkovChain(zs, pr, validate = true)

    merge(mcZi, mcZi)
end

# prepare the mapping X' = f(X,Z) which returns continous values but
# not necessary to locate on the grid points
ftest(X,Z) = begin
    
    # X,Z shall be vectors respectively

    # step: compute X' = f(X,Z)
    Xp = (X .+ Z) ./ 2

    return Xp
end

# run the algorithm; parallization is available
@time mcY = mmc.young2(
    ftest,
    xgrids,
    Zproc,
    parallel = false
)







#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Young (2010) non-stochastic simulation

--------------------
## Scenario: 

Controlled Markov chain Y := (X,Q,Z)

where:

Q  = h(X,Z)                 --> intra-period/implicit endogenous states
X' = g(X,Q,Z)               --> endogenous states 
Z  ~ exogenous Markov chain --> exogenous shock

and (g,h) are the (decision) rules that affects the transition matrix.

--------------------
## Notes:

- Any of X,Q,Z can be vector-valued.
- `Q` transition probability is _implicit_: controlled by decision rule `(h,g)`
  simultaneously and it is hard to directly write it. It is typically defined as
  something "intra-period".
- `Z` is the exogenous stochasticity that drives the randomness of the process.

--------------------
## In economics:

- Many general equilibrium models that involve multiple (types of) agents fit in
  this framework. Especially, the model features pricing mechanism that depends
  endogenously on the interaction of these agents. e.g. two agent models, TANK
- In such economies:
    - X: typical aggregate state variables such as capital, asset holding
    - Z: typical aggregate uncertainties such as TFP shock
    - Q: equilibrium variables that
        - depends on some endogenous stuffs (like hh's optimality conditions) 
          but without explicit solutions. Such as bond prices that are decided
          by the two agent's demand/supply endogenously. Such a price must clear
          the corresponding asset market, however, the clearing condition doesnt
          depend on the price explicitly.
        - maximizes some equilibrium objects such as an optimal tax policy that
          tries to maximizes the social welfare. Such a policy, by definition,
          must be jointly pinned down with the whole equilibrium.
- The multivariate Markov chain Y = (X,Q,Z) gives a full picture about how the
  aggregate dynamics evolves without hiding Q from the readers as what standard
  notations did. It is very helpful to numerical experiments.

--------------------
## Preparation for calling the API:

- A decision rule `(X',Q') = f(X,Q,Z,Z')` that deicdes how the endogenous states
  X and Q change across any two periods. The prime mark denotes "next period".
  Here Z' is required as Q' = h(X',Z'). Practically, you can wrap an interpolant
  or approximator with a function and pass it to the API.
- A collection of grids for X and Q, dimension by dimension.
- A MultivariateMarkovChain that describes the transition of Z shocks.

--------------------
## Illustrative example below:

- X: 2-vector in [0,1]^2
    - rule: Xp equals to the average of X, Q and Z (to introduce the joint depe-
      ndence on X, Q, and Z with the least effort)
- Z: 2-vector in [0,1]^2 
    - rule: follows a VAR(1); WLOG, assume cov(Z) = 0 
- Q: 2-vector in [0,1]^2
    - rule: the max and min among today's X and Z (to introduce the joint depen-
      dence on X and Z.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

# prepare grid spaces
xgrids = [
    LinRange(0,1, 10),
    LinRange(0,1, 10),
]
qgrids = [
    LinRange(0,1, 10),
    LinRange(0,1, 10),
]

# prepare the Z's process
Zproc = let

    # fake a AR(1) transition matrix for one dimension of Z
    pr = mmc.tauchen(
        3, # no. point per dimension of Z
        0.9, # AR(1) coefficient
        0.1, # Std Var of the error term
    ).probs
    zs = [[z,] for z in LinRange(0,1, pr |> size |> first)]

    mcZi = mmc.MultivariateMarkovChain(zs, pr, validate = true)

    merge(mcZi, mcZi)
end

# prepare the mapping (X',Q') = f(X,Q,Z,Zp) which returns continous values but
# not necessary to locate on the grid points
ftest(X,Q,Z,Zp) = begin
    
    # X,Q,Z,Zp shall be vectors respectively

    # step: compute X' = g(X,Q,Z)
    Xp = (X .+ Q .+ Z) ./ 3

    # step: compute Q' = h(X',Z')
    tmpMax = max(
        maximum(Xp),
        maximum(Zp),
    )
    tmpMin = min(
        minimum(Xp),
        minimum(Zp),
    )
    Qp = [tmpMax, tmpMin]

    # returns a tuple of two vector-like: Xp and Qp
    return Xp, Qp
end

# run the algorithm; parallization is available
@time mcY = mmc.young3(
    ftest,
    xgrids,
    qgrids,
    Zproc,
    parallel = true
)


