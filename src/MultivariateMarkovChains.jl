module MultivariateMarkovChains

using LinearAlgebra, SparseArrays
import Printf: @printf, @sprintf
import Statistics
import Random: MersenneTwister

using  StaticArrays             # for VAR1{D}
import MultivariateStats as mvs # for estimation
import SplitApplyCombine as sac # for convenient data manipulation
import Distributions as dst     # for VAR1{D} covariance structure



include("math.jl") # helpers

include("ar1.jl") # AR(1) & VAR(1) process representations

include("mc.jl") # multivariate Markov chain representation

include("fit.jl") # construct multivariate Markov chains from data

include("approx.jl") # approximate multivariate Markov chains from AR(1)/VAR(1)


end # MultivariateMarkovChains
