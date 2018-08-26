module Simulator
export randx, simulate
using Parameters
using Distributions

function randx(n;support=false)
    [[s₀ * (2*rand()-1) * 20000,
      s₀ * rand(LogNormal(0,3*ς)),
      (2*rand()-1) * 0.0050,
      support ? 0 : (2*rand()-1) * 2000,
      support ? 0 : (2*rand()-1) * 2000]
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
    [[S[i],P[i],r_a[i],r_b[i]] for i in 2:length(S)]
end

end
