# src/QuantumDD/DD_sequences.jl
using QuantumToolbox: QuantumObject
export get_pulse_times, get_modulation_function, get_shaped_pulses, get_control_terms, make_control_terms, control_function
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
    else
        error("Unsupported DD sequence: $sequence")
    end
end

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

function control_function(t::Float64, pulses::Vector{Pulse}, amp::Real=π)
    for pulse in pulses
        if pulse.start ≤ t ≤ pulse.stop
            if pulse.shape == "square"
                return amp/(pulse.stop - pulse.start)
            elseif pulse.shape == "gaussian"
                tc = (pulse.start + pulse.stop) / 2
                σ = (pulse.stop - pulse.start) / 6          # the length of the pulse contains 6σ ~ area of 99.7% of complete gaussian
                return amp * exp(-((t - tc)^2) / (2 * σ^2)) / (sqrt(2π) * σ)
            else
                error("Unknown pulse shape: $(pulse.shape)")
            end
        end
    end
    return 0.0
end
#--------------------------------------------------------------------------------------------------------------------------------------

function get_control_terms(pulses::Vector{Pulse}; pulse_amplitude=π)
    control_terms = []
    for axis in (:x, :y, :z)
        op = axis == :x ? sigmax() : axis == :y ? sigmay() : sigmaz()
        axis_pulses = filter(p -> p.axis == axis, pulses)
        for pulse in axis_pulses
            fn = (p, t) -> control_function(t, [pulse], pulse_amplitude)
            push!(control_terms, (op, fn))
        end
    end

    return control_terms
end
#---------------------------------------------------------------------------------------------------------------------------------------

function make_composite_control_term(pulses::Vector{Pulse}, operator;
                                     amp::Real = π)
    total_fn = (dummy_p,t) -> begin
        sum(control_function(t, [p], amp) for p in pulses)
    end
    return (operator, total_fn)
end
#---------------------------------------------------------------------------------------------------------------------------------------
# function make_composite_control_terms(pulses::Vector{Pulse}; amp::Real = π)
#     controls = []

#     for (axis, op) in zip((:x, :y, :z), (sigmax(), sigmay(), sigmaz()))
#         axis_pulses = filter(p -> p.axis == axis, pulses)
#         if !isempty(axis_pulses)
#             push!(controls, make_composite_control_term(axis_pulses, op; amp))
#         end
#     end

#     return controls
# end

function make_composite_control_terms(pulses::Vector{Pulse}; amp=π)
    grouped = Dict(:x => Pulse[], :y => Pulse[], :z => Pulse[])

    for pulse in pulses
        push!(grouped[pulse.axis], pulse)
    end

    terms = []
    for axis in (:x, :y, :z)
        pulses_axis = grouped[axis]
        if !isempty(pulses_axis)
            H = axis == :x ? sigmax() : axis == :y ? sigmay() : sigmaz()
            push!(terms, (H, (dummy_p,t) -> sum(control_function(t, [p], amp) for p in pulses_axis)))
        end
    end

    return terms
end

#--------------------------------------------------------------------------------------------------------------------------------------
"""
    make_control_terms(sequence::String, total_time::Float64, n_pulses::Int;
                       pulse_duration::Float64=0.05,
                       shape::String="square",
                       axis::String="X",
                       pulse_amplitude=π,
                       center_pulse::Bool=true)

Creates control terms suitable for `run_simulation`, wrapping the full process of:
1. Generating pulse times
2. Creating shaped `Pulse` structs
3. Building time-dependent control terms
(useful when simulating simple sequences like CPMG, UDD, Hahn echo)
Returns: Vector of (Operator, f(t)) tuples
"""
function make_control_terms(sequence::String, total_time::Float64, n_pulses::Int;
                            pulse_duration::Float64=0.05,
                            shape::String="square",
                            axis::Symbol=:x,
                            pulse_amplitude=π,
                            center_pulse::Bool=true)
    pulses = get_shaped_pulses(sequence, total_time, n_pulses;
                               pulse_duration=pulse_duration,
                               shape=shape,
                               axis=axis,
                               center_pulse=center_pulse)
    
    return make_composite_control_terms(pulses)
    #return get_control_terms(pulses; pulse_amplitude=pulse_amplitude)
end
