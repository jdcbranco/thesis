#include("./Parameters.jl")
#include("./Simulator.jl")
if something(findfirst(isequal("./"),LOAD_PATH),0) == 0
    push!(LOAD_PATH, "./")
end
using Distributed
using Distributions
using ForwardDiff
using JuMP
using NLopt
using GR
using DataFrames
using MultivariateStats
using GLMNet
using Flux:batch
using DelimitedFiles
using Statistics
using Parameters
using Simulator

#Basis functions
ϕ(x) = [1, v(x)/20000,-(q(x)+h(x))*v(x)/20000, -(v(x)^2)/1e6, p(x), q(x)^2, h(x)^2, q(x)+h(x)]#[v(x), v(x)^2, p(x), p(x)^2, q(x)^2 + h(x)^2, log(s(x)),
        #tanh(v(x)/1000), tanh((q(x)+h(x))/1000)]
# ϕ(x) = [v(x), y(x), s(x), p(x), q(x), h(x), s(x)^2, p(x)^2, q(x)^2 + h(x)^2, log(s(x)),
#                 s(x)*q(x), h(x)*s(x), h(x)*p(x), tanh(v(x)/1000), tanh((q(x)+h(x))/1000)]

#Function approximations
V(θ,x) =  sum(θ .* ϕ(x))
δᵃ(θ,x) = sum(θ .* ϕ(x))
δᵇ(θ,x) = sum(θ .* ϕ(x))
ξ(θ,x) =  sum(θ .* ϕ(x))

function get_param(X)
    x = X[1:5]
    θ = X[6:length(X)]
    x,θ
end

function get_x(X)
    X[1:5]
end

#Value funcion as function of one master argument
#with both the state x and the parameters θᵛ
function V(X)
    x,θ = get_param(X)
    V(θ,x)
end

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
∇V(θ,x) = ForwardDiff.gradient(V,[x;θ])
∇Vx(θ,x) = ∇V(θ,x)[1:5]
∇²V(θ,x) = ForwardDiff.hessian(V,[x;θ])
∇²Vx(θ,x) = ∇²V(θ,x)[1:5,1:5]
∂Vy(θ,x) = ∇V(θ,x)[1]
∂Vs(θ,x) = ∇V(θ,x)[2]
∂Vp(θ,x) = ∇V(θ,x)[3]
∂Vq(θ,x) = ∇V(θ,x)[4]
∂Vh(θ,x) = ∇V(θ,x)[5]
∂∂Vss(θ,x) = ∇²V(θ,x)[2,2]
∂∂Vpp(θ,x) = ∇²V(θ,x)[3,3]
∂∂Vsp(θ,x) = ∇²V(θ,x)[2,3]

#Value ingredients
Δᵃ(x,δᵃ,rᵃ) = [y(x) + (rᵃ>=δᵃ ? s(x)*(1+δᵃ) : 0), s(x) + s(x)*rᵃ, p(x) + s(x)*rᵃ, q(x) + (rᵃ>=δᵃ ? -1 : 0), h(x)]
Δᵇ(x,δᵇ,rᵇ) = [y(x) + (rᵇ>=δᵇ ? -s(x)*(1-δᵇ) : 0), s(x) - s(x)*rᵇ, p(x) - s(x)*rᵇ, q(x) + (rᵇ>=δᵇ ?  1 : 0), h(x)]
ΔᵃV(θ,x,δᵃ,rᵃ) = V(θ,Δᵃ(x,δᵃ,rᵃ)) - V(θ,x)
ΔᵇV(θ,x,δᵇ,rᵇ) = V(θ,Δᵇ(x,δᵇ,rᵇ)) - V(θ,x)
ΔᵃᵖV(θ,x,rᵖ) = cheap(x) ? V(θ,[y(x),s(x),p(x)+s(x)*rᵖ,q(x),h(x)]) - V(θ,x) : 0
ΔᵇᵖV(θ,x,rᵖ) = rich(x) ? V(θ,[y(x),s(x),p(x)-s(x)*rᵖ,q(x),h(x)]) - V(θ,x) : 0
EΔᵃV(θ,x,δᵃ) = mean([ΔᵃV(θ,x,δᵃ,r) for r in rᵃ(x)])
EΔᵇV(θ,x,δᵇ) = mean([ΔᵇV(θ,x,δᵇ,r) for r in rᵇ(x)])
EΔᵃᵖV(θ,x) = mean([ΔᵃᵖV(θ,x,r) for r in rᵃ(x)])
EΔᵇᵖV(θ,x) = mean([ΔᵇᵖV(θ,x,r) for r in rᵇ(x)])

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
function AV(θ,x,δᵃ,δᵇ)
    θ*(𝔭*s(x)-p(x))*∂Vp(θ,x)
    + 0.5 * s(x)^2 *(ς^2 * ∂∂Vss(θ,x) + κ^2 * ∂∂Vpp(θ,x) + 2 * ϱ * ς * κ * ∂∂Vsp(θ,x))
    + λᵃ^-1 * EΔᵃV(θ,x,δᵃ) + λᵇ^-1 * EΔᵇV(θ,x,δᵇ) + λᵖ^-1 * EΔᵃᵖV(θ,x) + λᵖ^-1 * EΔᵇᵖV(θ,x)
end

#Infinitesimal operator for R
function AR(x,δᵃ,δᵇ)
    Parameters.θ*(𝔭*s(x)-p(x))*∂Rp(x)
    + 0.5 * s(x)^2 *(ς^2 * ∂∂Rss(x) + κ^2 * ∂∂Rpp(x) + 2 * ϱ * ς * κ * ∂∂Rsp(x))
    + λᵃ^-1 * EΔᵃR(x,δᵃ) + λᵇ^-1 * EΔᵇR(x,δᵇ) + λᵖ^-1 * EΔᵃᵖR(x) + λᵖ^-1 * EΔᵇᵖR(x)
end

#L operator as described in the thesis
function LV(θ,x,δᵃ,δᵇ)
    AR(x,δᵃ,δᵇ) + AV(θ,x,δᵃ,δᵇ) - ρ*V(θ,x)
end

#Intervention function as described in the thesis
function Γ(x,ξ)
    x .+ [-(s(x)-p(x))*ξ - abs(ξ)*s(x)*χ, 0, 0, 0, ξ]
end

#Intervention operator
function MV(θ,x,ξ)
    V(θ,Γ(x,ξ))-V(θ,x)
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

#Greedy controls
maxδᵃ(θ,x) = 0.0001*argmax([EΔᵃV(θ,x,δ)+EΔᵃR(x,δ) for δ in 0.0001:0.0001:0.0020])
maxδᵇ(θ,x) = 0.0001*argmax([EΔᵇV(θ,x,δ)+EΔᵇR(x,δ) for δ in 0.0001:0.0001:0.0020])
maxξ(θ,x)  = validξ(q(x),h(x))[argmax([V(θ,Γ(x,ξ)) for ξ in validξ(q(x),h(x))])]

function V_estimate(θ,x)
    V(θ,x) + max(LV(θ,x,maxδᵃ(θ,x),maxδᵇ(θ,x)),MV(θ,x,maxξ(θ,x)))
    #R(x)
end

function total_reward(x₀;θᵃ=θᵃ,θᵇ=θᵇ,θʷ=θʷ,N=1000,T=1,Simulations=500,Safe=true)
    Δt = 1/N
    reward = 0
    reward = @distributed (+) for i = 1:Simulations
        reward_i = 0
        sim = simulate(x₀;N=N,T=T)
        t=1
        x = x₀
        for j in sim
            bid = Safe ? min(0.01,max(0.0001,δᵇ(x))) : δᵇ(θᵇ,x)
            offer = Safe ? min(0.01,max(0.0001,δᵃ(x))) : δᵃ(θᵃ,x)
            hedge = ξ(θʷ,x)
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


X_support = [randx(19;support=true);[x₀]]
ϕ_X_support = ϕ.(X_support)
ϕ_X_data_support = permutedims(batch(ϕ_X_support))
V_support = R.(X_support)

X_sample = randx(20)
ϕ_X = ϕ.(X_sample)
ϕ_X_data = permutedims(batch(ϕ_X))

#The learning rates below indicate how much % we want to learn from the current best
#and how much we want to learn from the new optimized function, given the new data
learning_rate = 0.25
learning_best = 0.50

#Initial values for the function approximations parameters
θᵛ = rand(length(ϕ(ones(5))))
θᵃ = zeros(length(ϕ(ones(5))))
θᵇ = zeros(length(ϕ(ones(5))))
θʷ = zeros(length(ϕ(ones(5))))

#The best reward is the best known expected total reward possible from state x₀
#we also keep track of the policies that generated the best reward
if !@isdefined best_reward
    best_θᵃ = θᵃ
    best_θᵇ = θᵇ
    best_θʷ = θʷ
    best_θᵛ = θᵛ
    best_reward = -Inf
end

function maxδᵃ(θ)
    mean([maxδᵃ(θ,x) for x in X_sample])
end

function maxδᵇ(θ)
    mean([maxδᵇ(θ,x) for x in X_sample])
end

function maxξ(θ)
    mean([maxξ(θ,x) for x in X_sample])
end

function V_estimate(θ)
    vcat([V_estimate(θ,x) for x in X_sample]...)
end

function V_bar(θ)
    vcat([V(θ,x) for x in X_sample]...)
end

function V_constraint1(θ)
    vcat([∇Vx(θ,x) .* ∇R(x) for x in X_sample]...)
end

# function second_order_condition(grad1,grad2)
#     [(grad1[i]-grad1[j])*(min(grad2[i]-grad2[j],grad2[i]+grad2[j])) for i in 1:5 for j in i:5 if i!=j]
# end
function second_order_condition(grad1,grad2)
    [grad1[i]*grad2[i] + grad1[j]*grad2[j] - abs(grad1[i]*grad2[j]+grad1[j]*grad2[i]) for i in 1:5 for j in i:5 if i!=j]
end

function third_order_condition(grad1, grad2)
    [grad1[i]*grad2[i] + grad1[j]*grad2[j] + grad1[k]*grad2[k] - abs(grad1[i]*grad2[j]+grad1[j]*grad2[i]) - abs(grad1[i]*grad2[k]+grad1[k]*grad2[i]) - abs(grad1[k]*grad2[j]+grad1[j]*grad2[k]) for i in 1:5 for j in i:5 for k in 1:5 if i<j && j<k]
end

function V_constraint2(θ)
    vcat([second_order_condition(∇Vx(θ,x),∇R(x)) for x in X_sample]...)
end

function V_constraint3(θ)
    vcat([third_order_condition(∇Vx(θ,x),∇R(x)) for x in X_sample]...)
end

#Lagrangian objective
function objective(z)
    θ = z[1:length(θᵛ)]
    λ₁ = z[length(θᵛ)+1:length(θᵛ)+5*length(X_sample)]
    λ₂ = z[length(θᵛ)+1+5*length(X_sample):length(θᵛ)+15*length(X_sample)]
    λ₃ = z[length(θᵛ)+1+15*length(X_sample):length(θᵛ)+25*length(X_sample)]
    mean((V_estimate(θ)-V_bar(θ)).^2) + 2* sum(abs.(θ)) #- λ₁'*V_constraint1(θ) - λ₂'*V_constraint2(θ) - λ₃'*V_constraint3(θ) #-sum(λ₁) -sum(λ₂) -sum(λ₃) #this is a penalty/regularization
end
objective(θ,λ₁,λ₂,λ₃) = objective([θ;λ₁;λ₂;λ₃])
∇objective(θ,λ₁,λ₂,λ₃) = ForwardDiff.gradient(objective,[θ;λ₁;λ₂;λ₃])[1:length(θᵛ)]

function minimize_lagrangian(θ₀,λ₁,λ₂,λ₃;gradient_step = 10^-10, n_iter = 40)
    F = zeros(n_iter)
    θ = [θ₀ for x in 1:n_iter]
    for iter = 1:n_iter
        F[iter] = objective(θ[iter],λ₁,λ₂,λ₃)
        if iter>1
            if F[iter]>F[iter-1]
                println("> F[$iter] is worse than F[$(iter-1)]: $(F[iter]) > $(F[iter-1])")
                return F[iter-1],θ[iter-1]
            elseif iter==n_iter
                return F[iter],θ[iter]
            elseif gradient_step < 10^-5
                gradient_step *= max(1,log(abs(F[iter])))
            end
        end
        θ[iter+1] = θ[iter] - gradient_step * ∇objective(θ[iter],λ₁,λ₂,λ₃)
        println("> Objective F(θ[$iter])): $(F[iter])")
    end
end

function solve_problem(θ₀;
    n_outer_iter = 20, n_inner_iter = 20, gradient_step = 10^-10, adjoint_step=10^-5)
    #Lagrange multipliers
    λ₁ = ones(5*length(X_sample))
    λ₂ = ones(10*length(X_sample))
    λ₃ = ones(10*length(X_sample))
    F = zeros(n_outer_iter)
    θ = [θ₀ for x in 1:n_outer_iter]
    for iter = 1:n_outer_iter
        println("Iteration $iter:")
        F[iter], θ[iter] = minimize_lagrangian(θ[iter],λ₁,λ₂,λ₃;gradient_step=gradient_step,n_iter=n_inner_iter)
        minλ = min([λ₁;λ₂;λ₃]...)
        c1 = V_constraint1(θ[iter])
        c2 = V_constraint2(θ[iter])
        c3 = V_constraint3(θ[iter])
        minc = min([c1;c2;c3]...)
        λ₁ += adjoint_step * minλ/minc * c1
        λ₂ += adjoint_step * minλ/minc * c2
        λ₃ += adjoint_step * minλ/minc * c3

        minλ_after = min([λ₁;λ₂;λ₃]...)
        λ₁ = max.(λ₁,zeros(length(λ₁)))
        λ₂ = max.(λ₂,zeros(length(λ₂)))
        λ₃ = max.(λ₃,zeros(length(λ₃)))
        println("minλ_before = $minλ, minλ_after = $minλ_after")
        println("Best F[$iter] = $(F[iter]) ")

        if iter>1
            if F[iter] > F[iter-1]
                println("F[$iter] is worse than F[$(iter-1)]: $(F[iter]) > $(F[iter-1])")
                return θ[iter-1]
            elseif iter==n_outer_iter
                return θ[iter]
            elseif adjoint_step < 0.05
                adjoint_step *= 2
            end
        end
        θ[iter+1] = θ[iter]
    end
end

θᵛ = rand(length(ϕ(ones(5))))
for iter = 1:10
    θᵛ = solve_problem(θᵛ,gradient_step = 10^-10, adjoint_step=0.00025)
    println("V(x₀) = ",V(θᵛ,x₀))
    #Replace half of the samples
    X_sample = [sample(X_sample,1); randx(2)]
    ϕ_X_data = permutedims(batch(ϕ.(X_sample)))
end

vfapolicy = DataFrame()
vfapolicy[:q] = vcat([y for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy[:h] = vcat([x for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy[:bid] = vcat([maxδᵇ(θᵛ,x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy[:offer] = vcat([maxδᵃ(θᵛ,x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy[:hedge] = vcat([maxξ(θᵛ,x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)

vfapolicy[:value] = vcat([V(θᵛ,x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy[:utility] = vcat([R(x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy[:wealth] = vcat([v(x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)

#titles3d("x", "y", "z")
surface(vfapolicy[:q],vfapolicy[:h],vfapolicy[:bid], title="bid")
surface(vfapolicy[:q],vfapolicy[:h],vfapolicy[:offer],  title="offer")
contour(vfapolicy[:q],vfapolicy[:h],vfapolicy[:hedge],  title="hedge")
surface(vfapolicy[:q],vfapolicy[:h],vfapolicy[:utility], title="utility")
surface(vfapolicy[:q],vfapolicy[:h],vfapolicy[:wealth], title="wealth")
surface(vfapolicy[:q],vfapolicy[:h],vfapolicy[:value], title="value")

#maxδᵃR(x) = 0.0001*argmax([EΔᵃR(x,δ) for δ in 0.0001:0.0001:0.0020])
#maxδᵇR(x) = 0.0001*argmax([EΔᵇR(x,δ) for δ in 0.0001:0.0001:0.0020])

# function U(x)
#     v(x) - s(x)^2*ϱ^2 * q(x)^2 - (s(x)-p(x))^2 * (ϱ^2 - κ^2) * h(x)^2
# end

# function U3(x)
#     y(x) + (q(x)*s(x) + h(x)*(s(x)-p(x)))*exp(-(q(x)*s(x) + h(x)*(s(x)-p(x)))*10^-6) - 0.5*(q(x)^2 + h(x)^2) - (q(x)+h(x))^2
# end
#
# contour(vfapolicy[:q],vfapolicy[:h],vcat([U3(x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...))
#
# contour(vfapolicy[:q],vfapolicy[:h],vcat([R(x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...))


#surface(vfapolicy[:q],vfapolicy[:h],vcat([EΔᵇV(θᵛ,x₀+[0,0,0,y,x],0.0001) for x in -2000:100:2000, y in -2000:100:2000]...))
#surface(vfapolicy[:q],vfapolicy[:h],vcat([EΔᵇR(x₀+[0,0,0,y,x],0.0001) for x in -2000:100:2000, y in -2000:100:2000]...))
#surface(vfapolicy[:q],vfapolicy[:h],vcat([maxδᵇR(x₀+[-100*(x+y),0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...), title="bid using R only")

#surface(vfapolicy[:q],vfapolicy[:h],vcat([maxδᵃ([3100,22662,-1.86],x₀+[0,0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...))


vfapolicy2 = DataFrame()
vfapolicy2[:q] = vcat([y for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy2[:h] = vcat([x for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy2[:bid] = vcat([maxδᵇ(θᵛ,x₀+[0,0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy2[:offer] = vcat([maxδᵃ(θᵛ,x₀+[0,0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy2[:value] = vcat([V(θᵛ,x₀+[0,0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy2[:utility] = vcat([R(x₀+[0,0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)
vfapolicy2[:wealth] = vcat([v(x₀+[0,0,0,y,x]) for x in -2000:100:2000, y in -2000:100:2000]...)

surface(vfapolicy2[:q],vfapolicy2[:h],vfapolicy2[:value], title="value2")
surface(vfapolicy2[:q],vfapolicy2[:h],vfapolicy2[:bid], title="bid2")
surface(vfapolicy2[:q],vfapolicy2[:h],vfapolicy2[:offer],  title="offer2")
surface(vfapolicy2[:q],vfapolicy2[:h],vfapolicy2[:utility], title="utility2")
surface(vfapolicy2[:q],vfapolicy2[:h],vfapolicy2[:wealth], title="wealth2")
#
# dfcash = DataFrame()
# dfcash[:cash0] = [maxδᵇ(θᵛ,x₀+[0,x,0,0,0]) for x in 10:1:200]
# dfcash[:cash1] = [maxδᵇ(θᵛ,x₀+[0,x,0,-10000,0]) for x in 10:1:200]
# plot(hcat(dfcash[:cash0],dfcash[:cash1]))
#

#
# [((EΔᵇV(θᵛ,x₀+[-100x,0,0,x,0],δ)+EΔᵇR(x₀+[-100x,0,0,x,0],δ),δ,x),EΔᵇV(θᵛ,x₀+[-100x,0,0,x,0],δ),EΔᵇR(x₀+[-100x,0,0,x,0],δ),δ,x) for δ in 0.0001:0.0001:0.0020, x in 1878:1879]
#
#
#
# [(EΔᵇV(θᵛ,x₀+[-100x,0,0,x,0],δ)+EΔᵇR(x₀+[-100x,0,0,x,0],δ),EΔᵇR(x₀+[-100x,0,0,x,0],δ),δ,x) for δ in 0.0001:0.0001:0.0020, x in 1878:1879]

# x2 = x₀+[-100*1879,0,0,1879,0]
# x1 = x₀+[-100*1878,0,0,1878,0]
# v(x1)>v(x2)
# v(Δᵇ(x1,0.0001,0.0001))>v(Δᵇ(x2,0.0001,0.0001))
# Δᵇ(x1,0.0001,0.0001),Δᵇ(x2,0.0001,0.0001)
# v(Δᵇ(x2,0.0001,0.0001))-v(x2)

##Value ingredients
#Δᵃ(x,δᵃ,rᵃ) = [y(x) + (rᵃ>=δᵃ ? s(x)*(1+δᵃ) : 0), s(x) + s(x)*rᵃ, p(x) + s(x)*rᵃ, q(x) + (rᵃ>=δᵃ ? -1 : 0), h(x)]
#Δᵇ(x,δᵇ,rᵇ) = [y(x) + (rᵇ>=δᵇ ? -s(x)*(1-δᵇ) : 0), s(x) - s(x)*rᵇ, p(x) - s(x)*rᵇ, q(x) + (rᵇ>=δᵇ ?  1 : 0), h(x)]
