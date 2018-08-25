using Distributions
using ForwardDiff
using JuMP
using NLopt
using Plots
#using StatPlots
using GR
using DataFrames
using MultivariateStats
using GLMNet
using Flux:batch

gr() #Use GR backend for Plots

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
funding_limit = -100 #cash limit we can borrow with ease
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
function v(x)
    y(x) + q(x)*s(x) + h(x)*(s(x)-p(x))
end

#Utility function
# function R(x)
#     v(x) - s(x)^2 * ς^2 * (q(x)+h(x))^2
# end

function R(x)
    y(x) + (q(x)*s(x) + h(x)*(s(x)-p(x)))*exp(-(q(x)*s(x) + h(x)*(s(x)-p(x)))*10^-6)
end

#Jump amplitude distributions
dist1 = Exponential(1/Λ)
dist2 = Exponential(1/Λᵖ)

quantiles1 = [quantile(dist1,i) for i in 0.0:0.02:0.99]
quantiles2 = [quantile(dist2,i) for i in 0.0:0.02:0.99]

rᵃ(x) = rich(x) ? quantiles2 : quantiles1
rᵇ(x) = cheap(x) ? quantiles2 : quantiles1

#basis functions
ϕ(x) = [v(x), y(x), s(x), p(x), q(x), h(x), s(x)^2, p(x)^2, q(x)^2, h(x)^2, log(s(x)),
        s(x)*q(x), h(x)*s(x), h(x)*p(x), tanh(v(x)/1000), tanh((q(x)+h(x))/1000)]

θᵛ = zeros(length(ϕ(ones(5))))
θᵃ = zeros(length(ϕ(ones(5))))
θᵇ = zeros(length(ϕ(ones(5))))
θʷ = zeros(length(ϕ(ones(5))))

#Function approximations
# V(x) =  sum(θᵛ .* ϕ(x))
# δᵃ(x) = sum(θᵃ .* ϕ(x))
# δᵇ(x) = sum(θᵇ .* ϕ(x))
# ξ(x) =  sum(θʷ .* ϕ(x))
V(x,θᵛ=θᵛ) =  sum(θᵛ .* ϕ(x))
δᵃ(x,θᵃ=θᵃ) = sum(θᵃ .* ϕ(x))
δᵇ(x,θᵇ=θᵇ) = sum(θᵇ .* ϕ(x))
ξ(x,θʷ=θʷ) =  sum(θʷ .* ϕ(x))

#Reward function partial derivatives
∇R(x) = ForwardDiff.gradient(R,x)
∇²R(x) = ForwardDiff.hessian(R,x)
∂Ry(x) = ∇R(x)[1]
∂Rs(x) = ∇R(x)[2]
∂Rp(x) = ∇R(x)[3]
∂Rq(x) = ∇R(x)[4]
∂Rh(x) = ∇R(x)[5]
∂∂Rss(x) = ∇²R(x)[2,2]
∂∂Rpp(x) = ∇²R(x)[3,3]
∂∂Rsp(x) = ∇²R(x)[2,3]

#Value function partial derivatives
∇V(x) = ForwardDiff.gradient(V,x)
∇²V(x) = ForwardDiff.hessian(V,x)
∂Vy(x) = ∇V(x)[1]
∂Vs(x) = ∇V(x)[2]
∂Vp(x) = ∇V(x)[3]
∂Vq(x) = ∇V(x)[4]
∂Vh(x) = ∇V(x)[5]
∂∂Vss(x) = ∇²V(x)[2,2]
∂∂Vpp(x) = ∇²V(x)[3,3]
∂∂Vsp(x) = ∇²V(x)[2,3]

#Value ingredients
Δᵃ(x,δᵃ,rᵃ) = [y(x) + (rᵃ>=δᵃ ? s(x)*(1+δᵃ) : 0), s(x) + s(x)*rᵃ, p(x) + s(x)*rᵃ, q(x) + (rᵃ>=δᵃ ? -1 : 0), h(x)]
Δᵇ(x,δᵇ,rᵇ) = [y(x) + (rᵇ>=δᵇ ? -s(x)*(1-δᵇ) : 0), s(x) - s(x)*rᵇ, p(x) - s(x)*rᵇ, q(x) + (rᵇ>=δᵇ ?  1 : 0), h(x)]
ΔᵃV(x,δᵃ,rᵃ) = V(Δᵃ(x,δᵃ,rᵃ)) - V(x)
ΔᵇV(x,δᵇ,rᵇ) = V(Δᵇ(x,δᵇ,rᵇ)) - V(x)
ΔᵃᵖV(x,rᵖ) = cheap(x) ? V([y(x),s(x),p(x)+s(x)*rᵖ,q(x),h(x)]) - V(x) : 0
ΔᵇᵖV(x,rᵖ) = rich(x) ? V([y(x),s(x),p(x)-s(x)*rᵖ,q(x),h(x)]) - V(x) : 0
EΔᵃV(x,δᵃ) = mean([ΔᵃV(x,δᵃ,r) for r in rᵃ(x)])
EΔᵇV(x,δᵇ) = mean([ΔᵇV(x,δᵇ,r) for r in rᵇ(x)])
EΔᵃᵖV(x) = mean([ΔᵃᵖV(x,r) for r in rᵃ(x)])
EΔᵇᵖV(x) = mean([ΔᵇᵖV(x,r) for r in rᵇ(x)])

#Reward ingredients
ΔᵃR(x,δᵃ,rᵃ) = R(Δᵃ(x,δᵃ,rᵃ)) - R(x)
ΔᵇR(x,δᵇ,rᵇ) = R(Δᵇ(x,δᵇ,rᵇ)) - R(x)
ΔᵃᵖR(x,rᵖ) = cheap(x) ? R([y(x),s(x),p(x)+s(x)*rᵖ,q(x),h(x)]) - R(x) : 0
ΔᵇᵖR(x,rᵖ) = rich(x) ? R([y(x),s(x),p(x)-s(x)*rᵖ,q(x),h(x)]) - R(x) : 0
EΔᵃR(x,δᵃ) = mean([ΔᵃR(x,δᵃ,r) for r in rᵃ(x)])
EΔᵇR(x,δᵇ) = mean([ΔᵇR(x,δᵇ,r) for r in rᵇ(x)])
EΔᵃᵖR(x) = mean([ΔᵃᵖR(x,r) for r in rᵃ(x)])
EΔᵇᵖR(x) = mean([ΔᵇᵖR(x,r) for r in rᵇ(x)])


#Infinitesimal operator for V
function AV(x,δᵃ,δᵇ)
    θ*(𝔭*s(x)-p(x))*∂Vp(x)
    + 0.5 * s(x)^2 *(ς^2 * ∂∂Vss(x) + κ^2 * ∂∂Vpp(x) + 2 * ϱ * ς * κ * ∂∂Vsp(x))
    + λᵃ^-1 * EΔᵃV(x,δᵃ) + λᵇ^-1 * EΔᵇV(x,δᵇ) + λᵖ^-1 * EΔᵃᵖV(x) + λᵖ^-1 * EΔᵇᵖV(x)
end

#Infinitesimal operator for R
function AR(x,δᵃ,δᵇ)
    θ*(𝔭*s(x)-p(x))*∂Rp(x)
    + 0.5 * s(x)^2 *(ς^2 * ∂∂Rss(x) + κ^2 * ∂∂Rpp(x) + 2 * ϱ * ς * κ * ∂∂Rsp(x))
    + λᵃ^-1 * EΔᵃR(x,δᵃ) + λᵇ^-1 * EΔᵇR(x,δᵇ) + λᵖ^-1 * EΔᵃᵖR(x) + λᵖ^-1 * EΔᵇᵖR(x)
end

function LV(x,δᵃ,δᵇ)
    AR(x,δᵃ,δᵇ) + AV(x,δᵃ,δᵇ) - ρ*V(x)
end

function Γ(x,ξ)
    x .+ [-(s(x)-p(x))*ξ - abs(ξ)*s(x)*χ, 0, 0, 0, ξ]
end

function MV(x,ξ)
    V(Γ(x,ξ))-V(x)
end

#Establishes valid values for ξ depending on the position
function validξ(q,h)
    if q+h> hedge_limit #long
        max(-q-h,-5):0
    elseif q+h < -hedge_limit
        0:min(-q-h,5)
    else
        0:0
    end
end

maxδᵃ(x) = 0.0001*argmax([EΔᵃV(x,δ)+EΔᵃR(x,δ) for δ in 0.0001:0.0001:0.0020])
maxδᵇ(x) = 0.0001*argmax([EΔᵇV(x,δ)+EΔᵇR(x,δ) for δ in 0.0001:0.0001:0.0020])
maxξ(x)  = validξ(q(x),h(x))[argmax([V(Γ(x,ξ)) for ξ in validξ(q(x),h(x))])]

function V_estimate(x)
    V(x) + max(LV(x,maxδᵃ(x),maxδᵇ(x)),MV(x,maxξ(x)))
end

function randx(n)
    [[s₀ * (2*rand()-1) * 20000,
      s₀ * rand(LogNormal(0,3*ς)),
      (2*rand()-1) * 0.0050,
      (2*rand()-1) * 2000,
      (2*rand()-1) * 2000]
    for i in 1:n]
end

function randx_support(n)
    [[s₀ * (2*rand()-1) * 20000,
      s₀ * rand(LogNormal(0,3*ς)),
      (2*rand()-1) * 0.0050,
      0,
      0]
    for i in 1:n]
end


function simulate(x₀;N=1000,T=1)
    Δt = 1/N
    dN_a = rand(Binomial(1,λᵃ*Δt),N*T)
    dN_b = rand(Binomial(1,λᵇ*Δt),N*T)
    r_a = rand(Exponential(1/Λ),N*T) .* dN_a
    r_b = rand(Exponential(1/Λ),N*T) .* dN_b
    dN_ap = rand(Binomial(1,λᵖ*Δt),N*T)
    dN_bp = rand(Binomial(1,λᵖ*Δt),N*T)
    r_ap = rand(Exponential(1/Λᵖ),N*T) .* dN_ap
    r_bp = rand(Exponential(1/Λᵖ),N*T) .* dN_bp
    dW_s = rand(Normal(0,sqrt(Δt)),N*T)
    dW_p = ϱ * dW_s + sqrt(1-ϱ^2) * rand(Normal(0,sqrt(Δt)),N*T)

    S = zeros(N*T)
    P = zeros(N*T)
    S[1] = s(x₀)
    P[1] = p(x₀)
    for i = 2:N*T
        is_cheap = cheap(P[i-1],S[i-1])
        is_rich = rich(P[i-1],S[i-1])
        r_a[i] *= (is_rich ? Λ/Λᵖ : 1)
        r_b[i] *= (is_cheap ? Λ/Λᵖ : 1)
        S[i] = S[i-1] + S[i-1] * (ς * dW_s[i] + r_a[i] - r_b[i])
        r_ap[i] *= is_cheap ? 1 : 0
        r_bp[i] *= is_rich ? 1 : 0
        P[i] = P[i-1] + S[i-1] * (κ * dW_p[i] + r_a[i] - r_b[i] + r_ap[i] - r_bp[i])
    end
    [[S[i],P[i],r_a[i],r_b[i],r_ap[i],r_bp[i]] for i in 2:length(S)]
end



function total_reward(x₀;N=1000,T=1,Simulations=500,Safe=true)
    Δt = 1/N
    reward = 0
    reward = @parallel (+) for i = 1:Simulations
        reward_i = 0
        sim = simulate(x₀;N=N,T=T)
        t=1
        x = x₀
        for j in sim
            bid = Safe ? min(0.01,max(0.0001,δᵇ(x))) : δᵇ(x)
            offer = Safe ? min(0.01,max(0.0001,δᵃ(x))) : δᵃ(x)
            hedge = ξ(x)
            if isnan(hedge)
                hedge=0
            end
            if isinf(hedge)
                hedge=0
            end
            new_x = [y(x), j[1], j[2], q(x), h(x)]
            new_rᵃ = j[3]
            new_rᵇ = j[4]
            if new_rᵃ >= offer
                new_x = new_x + [s(new_x),0,0,-1,0]
            end
            if new_rᵇ >= bid
                new_x = new_x + [-s(new_x),0,0,1,0]
            end
            if offer < 0 || bid < 0
                new_x = new_x + [- s(new_x)*χ,0,0,0,0]
            end
            posicao = q(x)+h(x)
            if posicao > hedge_limit
                hedge = max(-posicao,min(0,hedge))
            elseif posicao < -hedge_limit
                hedge = min(-posicao,max(0,hedge))
            end

            if abs(hedge) >= 1
                new_x = Γ(new_x,hedge)
            end
            reward_i += exp(-ρ*t*Δt)*R(new_x) - exp(-ρ*(t-1)*Δt)*R(x)
            if isnan(reward_i)
                println("NAN at ",x," -> ",new_x," hedge: ",hedge,", t: ",t)
                break
            end
            x = new_x
            t+=1
        end
        #reward += reward_i
        reward_i
    end
    reward/N
end


X_support = [randx_support(19);[x₀]]
ϕ_X_support = ϕ.(X_support)
ϕ_X_data_support = permutedims(batch(ϕ_X_support))
V_support = R.(X_support)

X_sample = randx(100)
ϕ_X = ϕ.(X_sample)
ϕ_X_data = permutedims(batch(ϕ_X))

learning_rate = 0.25
learning_best = 0.50

if !@isdefined best_reward
    best_θᵃ = θᵃ
    best_θᵇ = θᵇ
    best_θʷ = θʷ
    best_θᵛ = θᵛ
    best_reward = -Inf
end

for iter = 1:15
    δᵃ_data = maxδᵃ.(X_sample)
    δᵇ_data = maxδᵇ.(X_sample)
    ξ_data = maxξ.(X_sample)

    #new_θᵃlasso = glmnetcv(ϕ_X_data,δᵃ_data)
    #new_θᵃlasso_i = argmin(new_θᵃlasso.meanloss)
    #new_θᵃ = new_θᵃlasso.path.betas[:,new_θᵃlasso_i]
    new_θᵃ = ridge(ϕ_X_data,δᵃ_data,0.15;bias=false)
    new_δᵃ_data = [δᵃ(x,new_θᵃ) for x in X_sample]

    #new_θᵇlasso = glmnetcv(ϕ_X_data,δᵇ_data)
    #new_θᵇlasso_i = argmin(new_θᵇlasso.meanloss)
    #new_θᵇ = new_θᵇlasso.path.betas[:,new_θᵇlasso_i]
    new_θᵇ = ridge(ϕ_X_data,δᵇ_data,0.15;bias=false)
    new_δᵇ_data = [δᵇ(x,new_θᵇ) for x in X_sample]

    #new_θʷlasso = glmnetcv(ϕ_X_data,ξ_data)
    #new_θʷlasso_i = argmin(new_θʷlasso.meanloss)
    #new_θʷ = new_θʷlasso.path.betas[:,new_θʷlasso_i]
    new_θʷ = ridge(ϕ_X_data,ξ_data,0.15;bias=false)
    new_ξ_data = [ξ(x,new_θʷ) for x in X_sample]

    policy_error = sqrt(mean((new_δᵃ_data .- δᵃ_data) .^2)),  sqrt(mean((new_δᵇ_data .- δᵇ_data) .^2)), sqrt(mean((new_ξ_data .- ξ_data) .^2))
    #Update policy functions
    θᵃ = learning_best*best_θᵃ + (1-learning_rate-learning_best)*θᵃ + learning_rate * new_θᵃ
    θᵇ = learning_best*best_θᵇ + (1-learning_rate-learning_best)*θᵇ + learning_rate * new_θᵇ
    θʷ = learning_best*best_θʷ + (1-learning_rate-learning_best)*θʷ + learning_rate * new_θʷ

    function V_estimate_MC(x)
        total_reward(x)
    end

    V_data = V_estimate.(X_sample)
    #new_θᵛlasso = glmnetcv(ϕ_X_data,V_support)
    new_θᵛlasso = glmnetcv([ϕ_X_data;ϕ_X_data_support],[V_data;V_support])
    new_θᵛlasso_i = argmin(new_θᵛlasso.meanloss)
    new_θᵛ = new_θᵛlasso.path.betas[:,new_θᵛlasso_i]
    #new_θᵛ = ridge(ϕ_X_data,V_data,0.5;bias=false)
    new_V_data =  [V(x,new_θᵛ) for x in X_sample]

    V_error = sqrt(mean((new_V_data .- V_data) .^2))
    #Update value function
    θᵛ= (1-learning_rate-learning_best)*θᵛ + learning_best*best_θᵛ + learning_rate * new_θᵛ

    println("Iteration $(iter): policy_error is ",mean(policy_error)," while value_error is $(V_error)")

    current_reward = total_reward(x₀)
    println("Total reward: ",current_reward)

    if current_reward > best_reward
        best_θᵃ = θᵃ
        best_θᵇ = θᵇ
        best_θʷ = θʷ
        best_θᵛ = θᵛ
        best_reward = current_reward
        println("Found a candidate for best value, updating support.")
        #Update V_support
        V_support = total_reward.(X_support)
    end

    #New samples
    new_X_sample = randx(10)
    X_sample = [X_sample; new_X_sample]
    ϕ_X = [ϕ_X; ϕ.(new_X_sample)]
    ϕ_X_data = permutedims(batch(ϕ_X))
end

Rvalues  = [R(x₀+[-100i,0,0,i,0]) for i in -100000:10:100000]
#Rvalues  = [R(x₀+[-100*1000,i,0,1000,0]) for i in -10:0.01:10]
plot(Rvalues, label="utility")

Vvalues  = [V(x₀+[0,0,0,i,0]) for i in -100000:10:100000]
plot(Vvalues, label="value")

δᵃvalues = [δᵃ(x₀+[0,0,0,i,0]) for i in -2000:1:2000]
δᵇvalues = [δᵇ(x₀+[0,0,0,i,0]) for i in -2000:1:2000]
plot([-2000:1:2000],[δᵃvalues,δᵇvalues],label=["offer pfa","bid pfa"])

maxδᵃvalues = [maxδᵃ(x₀+[0,0,0,i,0]) for i in -2000:1:2000]
maxδᵇvalues = [maxδᵇ(x₀+[0,0,0,i,0]) for i in -2000:1:2000]
plot([-2000:1:2000],[maxδᵃvalues,maxδᵇvalues],label=["offer vfa","bid vfa"])

#Plot comparisons between PFA and VFA policies
plot([-2000:1:2000],[δᵃvalues,maxδᵃvalues],label=["offer pfa","offer vfa"])
plot([-2000:1:2000],[δᵇvalues,maxδᵇvalues],label=["bid pfa","bid vfa"])


ξvalues = [ξ(x₀+[0,0,0,i,0]) for i in -1000:1:1000]
plot(ξvalues, label="hedge pfa")

maxξvalues = [maxξ(x₀+[0,0,0,i,0]) for i in -2000:1:2000]
plot([-2000:1:2000],maxξvalues, label="hedge vfa")

# simulation1 = simulate(x₀)
# df = DataFrame()
# df[:s] = [x[1] for x in simulation1]
# df[:p] = [x[2] for x in simulation1]
# df[:r_a] = [x[3] for x in simulation1]
# df[:r_b] = [x[4] for x in simulation1]
# df[:r_ap] = [x[5] for x in simulation1]
# df[:r_bp] = [x[6] for x in simulation1]
# plot(df[:s],label="Price")


Vvalues1 = vcat([x for x in -2000:100:2000, y in -2000:100:2000]...)
Vvalues2 = vcat([y for x in -2000:100:2000, y in -2000:100:2000]...)
Vvalues3 = vcat([V(x₀+[0,0,0,x,y]) for x in -2000:100:2000, y in -2000:100:2000]...)
surface(Vvalues1,Vvalues2,Vvalues3)

# oldθᵃ=θᵃ
# oldθᵇ=θᵇ
# oldθʷ=θʷ
# oldθᵛ=θᵛ
#
# θᵃ=best_θᵃ
# θᵇ=best_θᵇ
# θʷ=best_θʷ
# θᵛ=best_θᵛ
