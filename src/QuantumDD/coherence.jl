using Interpolations, Base.Threads, Statistics, QuadGK, .Threads, QuantumToolbox, SparseArrays, ProgressMeter

export fast_average_fidelity_vs_time, χ, simulate_modulated_noise_fidelity, simulate_shaped_control_fidelity
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
    p = Progress(n_realizations, 1, "Simulating")
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
        ProgressMeter.next!(p)
    end 
    
    avg_fid = mean(fidelity_matrix, dims=1)[:]

    return T_vals, avg_fid
end

#--------------------------------------------------------------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------------------------------------------------------------------

"""
    simulate_modulated_noise_fidelity(; ψ₀, T_max, dt, n_realizations, target_std, dc, seed_offset, verbose, S_func, mod_func)

Simulates the fidelity decay of a single qubit subjected to modulated classical noise.

This function applies a time-dependent modulation to the noise using a user-defined modulation function `mod_func`,
typically used to simulate sequences like Hahn echo, CPMG, or UDD via piecewise constant toggling.

# Arguments
- `ψ₀`: Initial state of the qubit (defaults to equal superposition).
- `T_max`: Total simulation time.
- `dt`: Time step of the simulation.
- `n_realizations`: Number of noise realizations to average over.
- `target_std`: Target standard deviation for the generated noise.
- `dc`: DC offset in the noise (default: 0.0).
- `seed_offset`: Random seed offset (for reproducibility).
- `verbose`: Print progress every 10 realizations if true.
- `S_func`: Power spectral density function of the noise. **Must be provided.**
- `mod_func`: Time-dependent modulation function (defaults to `t -> 1.0`, i.e., no modulation = FID).

# Returns
- `T_vals`: Time points of the simulation.
- `avg_fid`: Average fidelity at each time point.

# Notes
- Noise is applied only along the x-axis (σₓ), modulated by `mod_func(t)`.
- Use `mod_func(t) = (-1)^n` step functions to represent toggling frames in DD sequences.

"""


function simulate_modulated_noise_fidelity(; 
    ψ₀=normalize(basis(2, 0) + basis(2, 1)),
    T_max::Float64=10.0,
    dt::Float64=0.01,
    n_realizations::Int=100,
    target_std=1.0,
    dc=0.0,
    seed_offset=0,
    verbose=false,
    S_func,
    mod_func::Function = t -> 1.0, # defaults to FID
    )
    if S_func === nothing
        error("S_func must be provided")
    end
    T_vals = collect(0:dt:T_max)
    n_T = length(T_vals)
    fidelity_matrix = zeros(Float32, n_realizations, n_T)
    @threads for i in 1:n_realizations
        noise_terms = []
        t_beta, β = generate_beta(S_func, T_max; dt=dt, target_std=target_std, seed=seed_offset + i + 1000, dc=dc)
        β_func = LinearInterpolation(t_beta, β, extrapolation_bc=Line())
        push!(noise_terms, (sigmaz(), (p,t) -> 0.5 * mod_func(t) * β_func(t)))

        tlist, result, fidelity_t = run_simulation(ψ₀; noise_terms=noise_terms, T=T_max, dt=dt)
        fidelity_matrix[i, :] .= Float32.(fidelity_t)

        if verbose && i % 10 == 0
            @info "Realization $i complete on thread $(threadid())"
        end
    end
    avg_fid = mean(fidelity_matrix, dims=1)[:]
    return T_vals, avg_fid
end

#--------------------------------------------------------------------------------------------------------------------------------------

"""
    simulate_shaped_control_fidelity(; ψ₀, T_max, dt, n_realizations, target_std, dc,
                                      seed_offset, verbose, S_func_x, S_func_y, S_func_z,
                                      control_terms)

Simulate fidelity decay under multiaxis colored noise with arbitrary shaped control pulses.

This function allows full specification of noise spectral densities in each axis and
accepts time-dependent control terms as precomputed `(Operator, Function)` tuples.
Each realization samples a random noise trajectory consistent with the given spectra.

# Arguments
- `ψ₀`: Initial pure state (default: `normalize(basis(2, 0) + basis(2, 1))`)
- `T_max`: Total evolution time
- `dt`: Time resolution
- `n_realizations`: Number of noise realizations
- `target_std`: Target standard deviation of noise amplitude
- `dc`: DC component of noise (default: 0.0)
- `seed_offset`: Offset to random seed per realization
- `verbose`: If true, logs progress every 10 realizations
- `S_func_x`, `S_func_y`, `S_func_z`: Spectral density functions per axis (optional)
- `control_terms`: Vector of `(Operator, Function)` pairs representing shaped controls

# Returns
A tuple `(T_vals, avg_fidelity)`:
- `T_vals`: Vector of times at which fidelity was computed
- `avg_fidelity`: Vector of average fidelity values over noise realizations

# Example
```julia
pulses = get_shaped_pulses("CPMG", 1.0, 4; shape="gaussian", axis="X")
controls = get_control_terms(pulses)

T, fid = simulate_shaped_control_fidelity(
    T_max=1.0,
    dt=1e-2,
    S_func_x=S_white,
    control_terms=controls,
    n_realizations=500
)
"""
#const ControlTerm = Tuple{QuantumObject{Operator, Dimensions{1, Tuple{Space}}, SparseMatrixCSC{ComplexF64, Int64}}, Function}

function simulate_shaped_control_fidelity(; 
    ψ₀ = normalize(basis(2, 0) + basis(2, 1)),
    T_max::Float64 = 10.0,
    dt::Float64 = 0.01,
    n_realizations::Int = 100,
    target_std_x = 0.0,
    target_std_y = 0.0,
    target_std_z = 0.1,
    dc = 0.0,
    seed_offset = 0,
    verbose = false,
    S_func_x = nothing,
    S_func_y = nothing,
    S_func_z = nothing,
    control_terms= []
)
    T_vals = collect(0:dt:T_max)
    n_T = length(T_vals)
    fidelity_matrix = zeros(Float32, n_realizations, n_T)
    p = Progress(n_realizations, 1, "Simulating")
    @threads for i in 1:n_realizations
        noise_terms = []

        if S_func_x !== nothing
            t_beta_x, β_x = generate_beta(S_func_x, T_max; dt=dt, target_std=target_std_x, seed=seed_offset + i + 1000, dc=dc)
            β_func_x = LinearInterpolation(t_beta_x, β_x, extrapolation_bc=Line())
            push!(noise_terms, (sigmax(), (p, t) -> 0.5 * β_func_x(t)))
        end

        if S_func_y !== nothing
            t_beta_y, β_y = generate_beta(S_func_y, T_max; dt=dt, target_std=target_std_y, seed=seed_offset + i + 2000, dc=dc)
            β_func_y = LinearInterpolation(t_beta_y, β_y, extrapolation_bc=Line())
            push!(noise_terms, (sigmay(), (p, t) -> 0.5 * β_func_y(t)))
        end

        if S_func_z !== nothing
            t_beta_z, β_z = generate_beta(S_func_z, T_max; dt=dt, target_std=target_std_z, seed=seed_offset + i + 3000, dc=dc)
            β_func_z = LinearInterpolation(t_beta_z, β_z, extrapolation_bc=Line())
            push!(noise_terms, (sigmaz(), (p, t) -> 0.5 * β_func_z(t)))
        end

        tlist, result, fidelity_t = run_simulation(ψ₀;
            noise_terms=noise_terms,
            control_terms=control_terms,
            T=T_max,
            dt=dt)

        fidelity_matrix[i, :] .= Float32.(fidelity_t)

        # if verbose && i % 10 == 0
        #     @info "Realization $i complete on thread $(threadid())"
        # end
        # if i % 20 == 0
        #     @info "Realization $i complete"
        # end
        ProgressMeter.next!(p)
    end

    avg_fid = mean(fidelity_matrix, dims=1)[:]
    return T_vals, avg_fid
end
