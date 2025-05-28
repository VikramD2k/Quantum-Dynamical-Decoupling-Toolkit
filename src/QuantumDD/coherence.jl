using Interpolations, Base.Threads, Statistics, CUDA, QuadGK, .Threads
export fast_average_fidelity_vs_time, χ, simulate_multiaxis_fidelity
"""
    fast_average_fidelity_vs_time(S_func; T_max, dt, n_realizations, target_std, seed_offset, use_gpu, verbose)

Compute average fidelity at final time for increasing T, under stochastic noise.

# Arguments
- `S_func`: Spectral density function ω → S(ω)
- `T_max`: Max total time
- `dt`: Time resolution
- `n_realizations`: Number of noise samples per T
- `target_std`: Std dev of β(t)
- `seed_offset`: Offset added to RNG seeds
- `use_gpu`: If true, allocate storage on GPU (not ODE solver)
- `verbose`: Print progress

# Returns
- `T_vals`: Time grid (dt:dt:T_max)
- `avg_fid`: Mean final fidelity at each time T
"""
function fast_average_fidelity_vs_time(S_func; 
                                       ψ₀=normalize(basis(2, 0) + basis(2, 1)),
                                       T_max::Float64=10.0,
                                       dt::Float64=0.01,
                                       n_realizations::Int=100,
                                       target_std=1.0,
                                       dc=0.0,
                                       seed_offset=0,
                                       use_gpu=true,
                                       verbose=false)

    T_vals = collect(0:dt:T_max)
    n_T = length(T_vals)

    # Allocate storage
    fidelity_matrix = zeros(Float32, n_realizations, n_T)

    @threads for i in 1:n_realizations
        t_beta, β_t = generate_beta(S_func, T_max; dt=dt, target_std=target_std , seed=seed_offset + i, dc=dc)
        β_func = LinearInterpolation(t_beta, β_t, extrapolation_bc=Line())  # or Flat() or Throw(), your choice

        noise_terms = [(sigmaz(), (p,t) -> 0.5 * β_func(t))]
        tlist, result, fidelity_t = run_simulation(ψ₀; noise_terms=noise_terms, T=T_max, dt=dt)
        #println("n_T = $n_T while fidelity_t has a length of $(length(fidelity_t))")
        fidelity_matrix[i, :] .= Float32.(fidelity_t)

        if  verbose && i % 10 == 0
            @info "Finished realization $i on thread $(threadid())"
        end
    end 
    
    avg_fid = mean(fidelity_matrix, dims=1)[:]

    return T_vals, avg_fid
end

#-------------------------------------------------------------------------------------------------------------

#using Filter_Functions.jl

"""
    χ(S_func, F_func, τ_vals; ω_min=1e-3, ω_max=100.0)

Compute the dephasing integral χ(τ) for a given spectral density S_func(ω)
and filter function F_func(ω, τ), across a vector of τ values.

Returns a vector of χ(τ) values.
"""
function χ(S_func::Function, F_func::Function, t_vals::AbstractVector, dt; truncate = true, parallelize = true)
    ω_min = 2π / t_vals[end]
    ω_max = π / dt
    χ_vals = zeros(length(t_vals))
    if parallelize
        # Preallocate results array
        @threads for i in eachindex(t_vals)
            t = t_vals[i]
            integrand(ω) = (only(S_func([ω])) / (ω^2 + 1e-12)) * F_func(ω, t)
            K = truncate ? quadgk(integrand, ω_min, ω_max)[1] :
                       quadgk(integrand, 0.0, Inf)[1]
            χ_vals[i] = K / π
        end
    else
        for i in eachindex(t_vals)
            t = t_vals[i]
            integrand(ω) = (only(S_func([ω])) / (ω^2 + 1e-12)) * F_func(ω, t)
            K = truncate ? quadgk(integrand, ω_min, ω_max)[1] :
                           quadgk(integrand, 0.0, Inf)[1]
            χ_vals[i] = K / π
        end
    end

    return χ_vals
end

#--------------------------------------------------------------------------------------------------
"""
    simulate_multiaxis_fidelity(; kwargs...) -> (T_vals, avg_fid)

Simulates decoherence under multi-axis time-dependent noise, with optional control pulses.

This function numerically evolves an initial quantum state ψ₀ under noisy Hamiltonians
involving stochastic noise along X, Y, and Z axes — each axis can be independently modulated
by user-defined spectral densities and control modulation functions. Optional shaped pulses
can be applied to simulate control operations (e.g., Hahn, CPMG).

Multiple stochastic noise realizations are simulated in parallel using multi-threading,
and the average state fidelity is computed as a function of total evolution time.

# Keyword Arguments
- `ψ₀::Ket`: Initial quantum state (default: equal superposition of |0⟩ and |1⟩).
- `T_max::Float64`: Total evolution time.
- `dt::Float64`: Time resolution for simulation.
- `n_realizations::Int`: Number of stochastic noise samples to average over.
- `target_std::Float64`: Desired standard deviation of β(t) noise realizations.
- `dc::Float64`: Optional DC offset added to all β(t) realizations.
- `seed_offset::Int`: Offset added to RNG seed for reproducibility.
- `use_gpu::Bool`: Placeholder for future GPU support (not used currently).
- `verbose::Bool`: Print progress every 10 realizations per thread.
- `S_func_x`, `S_func_y`, `S_func_z`: Spectral density functions ω → S(ω) for each axis.
- `mod_func_x`, `mod_func_y`, `mod_func_z`: Modulation functions t → ±1 for each axis (default: identity).
- `pulses::Vector`: Optional vector of shaped pulses (Dicts) for explicit control terms.

# Returns
- `T_vals::Vector`: Time values from 0 to T_max in steps of dt.
- `avg_fid::Vector`: Average state fidelity at each time T, computed over all realizations.

# Example
```julia
S = ω -> 1 / (ω^2 + 1)  # Lorentzian OU noise
mod = get_modulation_function(get_pulse_times("HAHN", 10.0, 1))

T_vals, fid = simulate_multiaxis_fidelity(
    S_func_z = S,
    mod_func_z = mod,
    T_max = 10.0,
    dt = 0.01,
    n_realizations = 1000
)
"""

function simulate_multiaxis_fidelity(; ψ₀=normalize(basis(2, 0) + basis(2, 1)),
    T_max::Float64=10.0,
    dt::Float64=0.01,
    n_realizations::Int=100,
    target_std=1.0,
    dc=0.0,
    seed_offset=0,
    use_gpu=true,
    verbose=false,
    S_func_x=nothing,
    S_func_y=nothing,
    S_func_z=nothing,
    mod_func_x::Function = t -> 1.0,
    mod_func_y::Function = t -> 1.0,
    mod_func_z::Function = t -> 1.0,
    pulses = nothing
    )

    T_vals = collect(0:dt:T_max)
    n_T = length(T_vals)
    fidelity_matrix = zeros(Float32, n_realizations, n_T)
    @threads for i in 1:n_realizations
        begin
            noise_terms = []

            if S_func_x !== nothing
                t_beta_x, β_x = generate_beta(S_func_x, T_max; dt=dt, target_std=target_std, seed=seed_offset + i + 1000, dc=dc)
                β_func_x = LinearInterpolation(t_beta_x, β_x, extrapolation_bc=Line())
                push!(noise_terms, (sigmax(), (p,t) -> 0.5 * mod_func_x(t) * β_func_x(t)))
            end

            if S_func_y !== nothing
                t_beta_y, β_y = generate_beta(S_func_y, T_max; dt=dt, target_std=target_std, seed=seed_offset + i + 2000, dc=dc)
                β_func_y = LinearInterpolation(t_beta_y, β_y, extrapolation_bc=Line())
                push!(noise_terms, (sigmay(), (p,t) -> 0.5 * mod_func_y(t) * β_func_y(t)))
            end

            if S_func_z !== nothing
                t_beta_z, β_z = generate_beta(S_func_z, T_max; dt=dt, target_std=target_std, seed=seed_offset + i + 3000, dc=dc)
                β_func_z = LinearInterpolation(t_beta_z, β_z, extrapolation_bc=Line())
                push!(noise_terms, (sigmaz(), (p,t) -> 0.5 * mod_func_z(t) * β_func_z(t)))
            end

            tlist, result, fidelity_t = run_simulation(ψ₀; noise_terms=noise_terms, T=T_max, dt=dt, pulses=pulses)
            fidelity_matrix[i, :] .= Float32.(fidelity_t)

            if verbose && i % 10 == 0
                @info "Realization $i complete on thread $(threadid())"
            end
        end
    end

    avg_fid = mean(fidelity_matrix, dims=1)[:]
    return T_vals, avg_fid
end