using QuantumToolbox
using LinearAlgebra
export control_function, run_simulation, simulate_rabi_with_z_noise


# function control_function(t, pulses, amp=π)
#     for pulse in pulses
#         if pulse.start ≤ t ≤ pulse.end
#             if pulse.shape == "square"
#                 return amp
#             elseif pulse.shape == "gaussian"
#                 tc = (pulse.start + pulse.stop) / 2
#                 σ = (pulse.stop - pulse.start) / 6
#                 return amp * exp(-((t - tc)^2) / (2σ^2))
#             end
#         end
#     end
#     return 0.0
# end

"""
    run_simulation(ψ₀, β_func; noise_terms=[], control_terms=[], pulses=nothing, T, dt, pulse_amplitude=π)

Evolve initial state ψ₀ under general time-dependent Hamiltonian.
Returns tlist, result, and fidelity vs ψ₀.

- `ψ₀`: initial state (Ket or DensityMatrix)
- `T`: total simulation time
- `dt`: time resolution
- `noise_terms`: list of (Op, func(t)) tuples
- `control_terms`: list of (Op, func(t)) tuples
- `pulses`: if given, generates control terms automatically
"""
function run_simulation(ψ₀;
                        noise_terms=[],
                        control_terms=[],
                        pulses=nothing,
                        T::Float64,
                        dt::Float64,
                        pulse_amplitude=π)

    tlist = 0:dt:T
    H_t = tuple(control_terms..., noise_terms...)
    result = mesolve(H_t, ψ₀, tlist; progress_bar=Val(false))
    #print(typeof(result.states[1]))
    fidelity_t = [real(only(ψ₀.data' * ψ.data * ψ₀.data)) for ψ in result.states]

    return tlist, result, fidelity_t
end

#-------------------------------------------------------------------------------------------------------

function simulate_rabi_with_z_noise(Ω, ω_drive, β_t, tlist; ψ₀=normalize(basis(2, 0) + basis(2, 1)))
    

    # Interpolation of noise
    β_func = LinearInterpolation(tlist, β_t)

    # Pauli matrices
    σx, σy, σz = sigmax(), sigmay(), sigmaz()

    # Time-dependent Hamiltonian
    H_t = [
        (σz, (p, t) -> 0.5 * β_func(t)),
        (σx, (p, t) -> Ω * cos(ω_drive * t)),
        (σy, (p, t) -> Ω * sin(ω_drive * t))
    ]
    H_t = tuple(H_t...) # Convert to tuple
    # Run simulation
    tlist_out, result, fid = run_simulation(ψ₀; noise_terms=H_t, T=maximum(tlist), dt=tlist[2] - tlist[1])
    return tlist_out, result, fid
end
