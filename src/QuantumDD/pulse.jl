# src/QuantumDD/DD_sequences.jl
export get_pulse_times, get_modulation_function, get_shaped_pulses
export generate_cdd_pulse_times, plot_pulse_schedule, plot_modulation

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
    elseif sequence == "PDD"
        return [(j + 1) * total_time / (n_pulses + 1) for j in 0:(n_pulses - 1)]
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
        left = generate_cdd_pulse_times(level - 1; t_start = t_start, t_end = t_mid)
        right = generate_cdd_pulse_times(level - 1; t_start = t_mid, t_end = t_end)
        return sort(vcat(left, [t_mid], right))
    end
end
#--------------------------------------------------------------------------------------------------------------------------------------
using Plots

"""
    plot_pulse_schedule(pulses::Vector{Pulse}; height=1.0, color=:black, y_offset=0.0)

Plot a visual schedule of the pulse sequence as vertical bars.
Each bar spans the pulse duration, color-coded by axis (x, y, z).

Arguments:
- `pulses`: Vector of Pulse objects (with t_start, t_end, shape, axis)
- `height`: Height of each pulse bar
- `color`: Base color or `nothing` to auto-select by axis
- `y_offset`: Vertical shift of the bar if overlaying on other plots
"""
function plot_pulse_schedule(pulses::Vector{Pulse}, total_time; height=π, color=:black, sequence="")
    t = Float64[]
    y = Float64[]
    push!(t, 0.0); push!(y, 0.0)
    
    for pulse in pulses
        t_start = pulse.start
        t_end = pulse.stop
        global t_duration = t_end - t_start
        push!(t, t_start); push!(y, 0.0)  # stay low before pulse
        push!(t, t_start); push!(y, height/t_duration)  # jump up
        push!(t, t_end); push!(y, height/t_duration)  # stay high
        push!(t, t_end); push!(y, 0.0)       # drop down
    end

    push!(t, total_time); push!(y, 0.0)  # finish at end

    plot(t, y, color=color,label = "$sequence", linewidth=2, linetype=:steppre, xlim=(0, total_time), xticks = 0:total_time/10:total_time, yticks = 0:(height/(5*t_duration)):height/t_duration, xlabel="Time", ylabel="Amplitude", title="$sequence Pulse Schedule")
end

"""
    plot_modulation(pulse_times::Vector{<:Real}, total_time;
                    color=:blue, sequence="")

Draw the ±1 modulation function y(t) that flips sign at each
time in `pulse_times`.  Useful for visualising CPMG, Hahn, UDD, etc.

Arguments
---------
- `pulse_times` : Vector of Float64 (or Real) giving the flip instants.
                  They will be sorted internally.
- `total_time`  : Duration T of the experiment.

Keyword options
---------------
- `color`       : Line colour (default `:blue`)
- `sequence`    : Optional label / title string.
"""
function plot_modulation(pulse_times::Vector{<:Real}, total_time;
                         color=:blue, sequence="")
    # Ensure ascending order and drop times beyond total_time, if any
    pulse_times = sort(filter(t -> t ≤ total_time, pulse_times))

    # Build staircase points
    t = Float64[]   # x-coordinates
    y = Int[]       # y-coordinates (±1)

    current_y = 1         # y(0) = +1 by convention
    push!(t, 0.0);  push!(y, current_y)

    for tp in pulse_times
        # stay at current value until tp
        push!(t, tp); push!(y, current_y)
        # flip sign at tp
        current_y *= -1
        push!(t, tp); push!(y, current_y)
    end

    # extend to total_time
    push!(t, total_time);  push!(y, current_y)

    # Plot
    plot(t, y;
         label     = sequence == "" ? false : sequence,
         linewidth = 2,
         linetype  = :steppost,
         color     = color,
         xlim      = (0, total_time),
         ylim      = (-1.2, 1.2),
         yticks    = (-1:1:1),
         xticks    = 0:total_time/10:total_time,
         xlabel    = "Time",
         ylabel    = "y(t)",
         title     = sequence == "" ? "Modulation Function" : "$sequence Modulation")
end