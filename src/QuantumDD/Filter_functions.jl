# src/QuantumDD/Filter_functions.jl
export filter_fid_vec, filter_fid, filter_cpmg, filter_udd

using LinearAlgebra, FFTW

# FID: F(ωτ) = 4 sin²(ωτ/2) / ω²
function filter_fid(ω::Real, τ::Real)
    return 4 * sin(0.5 * ω * τ)^2 / (ω^2 + 1e-12)
end

function filter_fid(ω::AbstractVector, τ::Real)
    return @. 4 * sin(0.5 * ω * τ)^2 / (ω^2 + 1e-12)
end

# CPMG (approximate closed form)
function filter_cpmg(ω::AbstractVector, τ::Real, n::Int)
    Δt = τ / n
    denom = @. ω^2 + 1e-12
    return @. (4 / denom) * sin(0.5 * ω * τ)^2 * sin(0.5 * ω * Δt)^2
end

# UDD: Generate pulse times
function generate_udd_pulse_times(n::Int, τ::Real)
    return [τ * sin(pi * j / (2n + 2))^2 for j in 1:n]
end

# UDD: Create modulation function
function generate_modulation_function(pulse_times::Vector{Float64}, τ::Real, num_points::Int=10_000)
    t = range(0, τ, length=num_points)
    y = ones(Float64, num_points)
    for pt in pulse_times
        y[t .>= pt] .*= -1
    end
    return t, y
end

# UDD: Numerical FFT-based filter
function filter_udd(ω::AbstractVector, τ::Real, n::Int, num_points::Int=10_000)
    pulse_times = generate_udd_pulse_times(n, τ)
    t, y_t = generate_modulation_function(pulse_times, τ, num_points)
    dt = t[2] - t[1]
    # y_omega = ∫ y(t) e^{iωt} dt
    y_omega = [sum(y_t .* exp.(1im * ωj * t)) * dt for ωj in ω]
    return abs2.(y_omega)
end
