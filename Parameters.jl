module Parameters
using Distributions
export v, y, s, p, q, h, rich, cheap, R, rᵃ, rᵇ
export ς,κ,λᵃ,λᵇ,λᵖ,Λ,Λᵖ,ϱ,s₀,p₀,𝔭,x₀,ρ,χ,hedge_limit

θ = 0.05 #speed of reversion to mean in OU process
ς = 0.02 #ETF vol
κ = 0.00008 #Premium vol
λᵃ = 500
λᵇ = 500
λᵖ = 500
Λ = 5000
Λᵖ = 10000
ϱ = 0.5 #correlation between ETF and Premium
s₀ = 100.0 #intiial price
p₀ = 0.0 #initial premium
𝔭 = 0.0 #fair premium
x₀ = [0.0, s₀, p₀, 0.0, 0.0] #initial state
ρ = 0.2 #reward discounting rate
χ = 0.0001 #transaction cost
hedge_limit = 1 #do not hedge if net position is less than 1
# State utils
y(x) = x[1]
s(x) = x[2]
p(x) = x[3]
q(x) = x[4]
h(x) = x[5]
cheap(p,s) = p <= -0.005*s
rich(p,s) = p >= 0.005*s
cheap(x) = cheap(p(x),s(x))
rich(x) = rich(p(x),s(x))
#Wealth function
v(x) =  y(x) + q(x)*s(x) + h(x)*(s(x)-p(x))
#Utility function
# function R(x)
#     y(x) + (q(x)*s(x) + h(x)*(s(x)-p(x)))*exp(-(q(x)*s(x) + h(x)*(s(x)-p(x)))*10^-6)
# end
function R(x)
    v(x)
end

#Jump amplitude distributions
dist1 = Exponential(1/Λ)
dist2 = Exponential(1/Λᵖ)

quantiles1 = [quantile(dist1,i) for i in 0.0:0.02:0.99]
quantiles2 = [quantile(dist2,i) for i in 0.0:0.02:0.99]

rᵃ(x) = rich(x) ? quantiles2 : quantiles1
rᵇ(x) = cheap(x) ? quantiles2 : quantiles1

end
