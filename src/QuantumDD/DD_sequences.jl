# src/QuantumDD/DD_sequences.jl
export get_pulse_times, get_modulation_function, get_shaped_pulses

function get_pulse_times(sequence::String, total_time::Float64, n_pulses::Int)
    sequence = uppercase(sequence)
    if sequence == "CPMG"
        return [(i + 0.5) * total_time / n_pulses for i in 0:(n_pulses - 1)]
    elseif sequence == "UDD"
        return [total_time * sin((π * (j + 1)) / (2 * n_pulses + 2))^2 for j in 0:(n_pulses - 1)]
    elseif sequence == "HAHN"
        return [total_time / 2]
    else
        error("Unsupported DD sequence: $sequence")
    end
end

function get_modulation_function(pulse_times::Vector{Float64})
    pulse_times = sort(pulse_times)
    return t -> (-1)^count(ti -> ti < t, pulse_times)
end

function get_shaped_pulses(sequence::String, total_time::Float64, n_pulses::Int;
                           pulse_duration::Float64=0.05,
                           shape::String="square",
                           axis::String="X",
                           center_pulse::Bool=true)
    pulse_times = get_pulse_times(sequence, total_time, n_pulses)
    pulses = []

    for t0 in pulse_times
        t_start = center_pulse ? t0 - pulse_duration/2 : t0
        t_end   = center_pulse ? t0 + pulse_duration/2 : t0 + pulse_duration
        push!(pulses, Dict(:start => t_start, :end => t_end, :shape => shape, :axis => axis))
    end

    return pulses
end
