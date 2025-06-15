# src/QuantumDD/Filter_functions.jl
# This file contains filter functions for quantum dynamical decoupling (QDD) sequences
export filter_fid_vec, filter_fid, filter_cpmg, filter_udd, filter_cdd
using LinearAlgebra, FFTW

#-------------------------------------------------------------------------------------------------------------------------------------

# FID: F(ωτ) = 4 sin²(ωτ/2) / ω²
function filter_fid(ω::Real, τ::Real)
    return 4 * sin(0.5 * ω * τ)^2 / (ω^2 + 1e-12)
end

#-------------------------------------------------------------------------------------------------------------------------------------

function filter_fid(ω::AbstractVector, τ::Real)
    return @. 4 * sin(0.5 * ω * τ)^2 / (ω^2 + 1e-12)
end

#-------------------------------------------------------------------------------------------------------------------------------------

# CPMG (approximate closed form for pulses of 0 duration):
function filter_cpmg(ω::AbstractVector, τ::Real, n_pulses::Int)
    Δt = τ / n_pulses
    denom = @. ω^2 + 1e-12
    return @. (4 / denom) * sin(0.5 * ω * τ)^2 * sin(0.5 * ω * Δt)^2
end

#-------------------------------------------------------------------------------------------------------------------------------------

function filter_udd(ω::AbstractVector, τ::Real, n::Int, num_points::Int=10_000)
    pulse_times = get_pulse_times("UDD", τ, n)
    y = get_modulation_function(pulse_times)  # symbolic modulation function y(t)

    t = range(0, τ, length=num_points)
    dt = t[2] - t[1]
    y_t = [y(ti) for ti in t]  # sample the symbolic function

    # Compute the numerical Fourier transform
    y_omega = [sum(y_t .* exp.(1im * ωj * t)) * dt for ωj in ω]
    return abs2.(y_omega)
end

#-------------------------------------------------------------------------------------------------------------------------------------

function filter_cdd(ω::AbstractVector, τ::Real, level::Int; num_points::Int=10_000)
    pulse_times = τ .* generate_cdd_pulse_times(level)
    t, y_t = generate_modulation_function(pulse_times, τ, num_points) # numerical modulation function y(t)
    dt = t[2] - t[1]
    exponential_matrix = exp.(1im .* ωj .* t)  # outer product for efficiency
    y_omega = exponential_matrix * y_t .* dt
    return abs2.(y_omega)
end

#-------------------------------------------------------------------------------------------------------------------------------------

function filter_pdd(ω::AbstractVector, τ::Real, n_pulses::Int; num_points::Int=10_000)
    pulse_times = get_pulse_times("PDD", τ, n_pulses)
    y = get_modulation_function(pulse_times)  # symbolic modulation function y(t)

    t = range(0, τ, length=num_points)
    dt = t[2] - t[1]
    y_t = [y(ti) for ti in t]  # sample the symbolic function

    # Compute the numerical Fourier transform
    exponential_matrix = exp.(1im .* ωj .* t)  # outer product for efficiency
    y_omega = exponential_matrix * y_t .* dt
    return abs2.(y_omega)
end