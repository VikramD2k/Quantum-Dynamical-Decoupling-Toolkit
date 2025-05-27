# src/QuantumDD/beta_generator.jl
export generate_beta, normalize_spectrum

using FFTW, Random, Statistics

"""
    normalize_spectrum(S_shape::Function, ωs::AbstractVector, σ::Real)

Given an unnormalized PSD shape `S_shape(ω)`,obtained from S(...) function in noise_models.jl, compute a properly scaled PSD `S_actual(ω)`
such that the total area under the curve over `ωs` is equal to `σ²`.

# Arguments
- `S_shape`: Function ω → S(ω) representing the shape of the PSD
- `ωs`: Vector of angular frequencies (positive values, typically from fftfreq)
- `σ`: Target standard deviation of the time-domain noise β(t)

# Returns
- A function `S_actual(ω)` that is properly normalized
"""
function normalize_spectrum(S_shape::Function, ωs::AbstractVector, σ::Real)
    Δω = ωs[2] - ωs[1]  # assumes uniform spacing
    S_vals = S_shape(ωs) 
    #S_vals = [S_shape(ω) for ω in ωs]
    # Clean up and sanitize
    S_vals[.!isfinite.(S_vals)] .= 0.0
    S_vals = clamp.(S_vals, 0.0, Inf)

    power = sum(S_vals) * Δω # Power = ∫ S(ω) dω
    if power == 0.0
        error("Power spectral density is zero, cannot normalize.")
    end
    scaling = σ^2 / power

    return ω -> scaling * S_shape(ω)
end

"""
    generate_beta(S_shape, T; dt=0.01, target_std=0.1, ...)

Generate a stationary real-valued stochastic process β(t) with duration `T` and time resolution `dt`, using inverse FFT synthesis.
The power spectral density `S_shape(ω)` is scaled internally to match the desired time-domain standard deviation `target_std`.

Returns a time vector and a noise vector with variance ≈ `target_std²`.
"""

function generate_beta(S_func, T; dt=0.01, target_std=0.1, seed=nothing,
                        oversample_factor=10, margin_factor=2.0, dc=0.0)
    if seed !== nothing
        Random.seed!(seed)
    end

    T_long = oversample_factor * T
    N = Int(round(T_long / dt))
    ωs = fftfreq(N, dt) .* 2π
    Δω = 2π / (N * dt)

    S_actual = normalize_spectrum(S_func, ωs, target_std)
    S_vals = S_actual(abs.(ωs))
    #S_vals = S_func(abs.(ωs))
    S_vals[.!isfinite.(S_vals)] .= 0.0
    S_vals[abs.(ωs) .< 1e-10] .= 0.0
    S_vals = clamp.(S_vals, 0.0, Inf)

    sqrt_input = S_vals .* Δω
    sqrt_input[.!isfinite.(sqrt_input)] .= 0.0
    sqrt_input = clamp.(sqrt_input, 0.0, Inf)
    magnitude = sqrt.(sqrt_input)

    phases = 2π * rand(N)
    spectrum = magnitude .* exp.(1im .* phases)
    spectrum[1] = 0.0
    if iseven(N)
        spectrum[N÷2 + 1] = real(spectrum[N÷2 + 1])
    end

    β_long = real(ifft(spectrum)) .* N
    β_long .-= mean(β_long)
    β_long .+= dc

    # Slice
    N_T = Int(round(T / dt))
    margin_pts = Int(round(margin_factor * N_T))
    start_idx = rand(margin_pts:(N - margin_pts - N_T))
    β_t = β_long[start_idx : start_idx + N_T - 1]
    tlist = collect(0:dt:(T - dt))

    return tlist, β_t
end

