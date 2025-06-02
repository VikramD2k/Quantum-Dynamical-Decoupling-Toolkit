# src/QuantumDD/DD_sequences.jl
export get_pulse_times, get_modulation_function, get_shaped_pulses, generate_cdd_pulse_times

struct Pulse
    start::Float64
    stop::Float64
    shape::String
    axis:: Symbol  # :x, :y, or :z
end

#--------------------------------------------------------------------------------------------------------------------------------------

function get_pulse_times(sequence::String, total_time::Float64, n_pulses::Int)
    sequence = uppercase(sequence)
    if sequence == "CPMG"
        return [(i + 0.5) * total_time / n_pulses for i in 0:(n_pulses - 1)]
    elseif sequence == "UDD"
        return [total_time * sin((π * (j + 1)) / (2 * n_pulses + 2))^2 for j in 0:(n_pulses - 1)]
    elseif sequence == "HAHN"
        return [total_time / 2]
    elseif sequence == "CDD"
        # Interpriting n_pulses as the level of CDD
        @assert n_pulses >= 0 "n_pulses must be non-negative for CDD"
        return total_time * generate_cdd_pulse_times(n_pulses)
    else
        error("Unsupported DD sequence: $sequence")
    end
end

#--------------------------------------------------------------------------------------------------------------------------------------

function get_modulation_function(pulse_times::Vector{Float64})
    pulse_times = sort(pulse_times)
    return t -> (-1)^count(ti -> ti < t, pulse_times)
end

#--------------------------------------------------------------------------------------------------------------------------------------

function get_shaped_pulses(sequence::String, total_time::Float64, n_pulses::Int;
                           pulse_duration::Float64=0.05,
                           shape::String="square",
                           axis::Symbol=:x,
                           center_pulse::Bool=true)::Vector{Pulse}
    @assert axis in (:x, :y, :z) "Unsupported axis: $axis"

    pulse_times = get_pulse_times(sequence, total_time, n_pulses)
    pulses = Pulse[]

    for t0 in pulse_times
        t_start = center_pulse ? t0 - pulse_duration/2 : t0
        t_end   = center_pulse ? t0 + pulse_duration/2 : t0 + pulse_duration
        push!(pulses, Pulse(t_start, t_end, shape, axis))
    end

    return pulses
end

#--------------------------------------------------------------------------------------------------------------------------------------

# CDD: Recursive modulation function over time interval [0, τ]
function generate_cdd_pulse_times(level::Int; t_start::Float64 = 0.0, t_end::Float64 = 1.0)
    if level == 0
        return Float64[]
    else
        t_mid = 0.5 * (t_start + t_end)
        left = generate_cdd_pulse_times(level - 1; t_start, t_mid)
        right = generate_cdd_pulse_times(level - 1; t_mid, t_end)
        return sort(vcat(left, [t_mid], right))
    end
end