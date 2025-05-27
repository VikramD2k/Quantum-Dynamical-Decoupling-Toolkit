# src/QuantumDD/Plotting.jl
using QuantumToolbox: sigmaz, sigmax, sigmay, expect
using Plots: plot, plot!, title!, xlabel!, ylabel!
export plot_spectrum

using FFTW, Plots, Printf

"""
    plot_spectrum(β_t, dt; log=true, normalized=true, label="β(t)", skip=1)

Plot the power spectrum of a real-valued time-domain signal.

# Arguments
- `β_t`: Noise time series
- `dt`: Time resolution
- `log`: If true, use log-log axes
- `normalized`: Normalize FFT by length for proper power interpretation
- `label`: Label for plot
- `skip`: Number of initial frequency bins to skip (e.g. to avoid DC)

# Returns
- Plot object (can be saved or displayed)
"""
function plot_spectrum(β_t, dt; log=true, normalized=true, label="β(t)", skip=1)
    N = length(β_t)
    fft_result = normalized ? fft(β_t) / N : fft(β_t)
    power = abs.(fft_result).^2

    half = 1:div(N, 2)
    freqs = (0:N-1) .* (1 / (N * dt))

    x = freqs[half][skip:end]
    y = power[half][skip:end]

    if log
        x = log10.(x)
        y = log10.(y)
        p = plot(x, y, label=label, xlabel="log₁₀(frequency)", ylabel="log₁₀(power)",
                 title="Power Spectrum of $label", lw=1.5)
    else
        p = plot(x, y, label=label, xlabel="Frequency", ylabel="Power",
                 title="Power Spectrum of $label", lw=1.5)
    end

    return p
end


#-------------------------------------------------------------------------------------------------------------

export plot_population

"""
    plot_population(tlist, states; basis=:z, title="Population vs Time")

Plot qubit population vs. time for a given measurement basis (σz by default).

# Arguments
- `tlist`: Time points
- `states`: Vector of state objects (Ket or DensityMatrix)
- `basis`: Measurement axis (:x, :y, :z)
- `title`: Title of the plot

# Returns
- Plot object
"""

function plot_population(tlist, states; basis=:z, title="Population vs Time")
    N = length(states)
    pop = zeros(N)

    # Choose measurement operator
    σ = basis == :z ? sigmaz() :
        basis == :x ? sigmax() :
        basis == :y ? sigmay() :
        error("Invalid basis: choose :x, :y, or :z")

    # Compute ⟨σ⟩ and map to [0,1] population
    for i in 1:N
        pop[i] = real(expect(σ, states[i])) / 2 + 0.5
    end

    p = plot(tlist, pop, label="P(|1⟩)", xlabel="Time", ylabel="Population", lw=2)
    title!(p, title)
    #grid!(p)
    return p
end

