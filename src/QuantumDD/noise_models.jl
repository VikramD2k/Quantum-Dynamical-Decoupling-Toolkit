# src/QuantumDD/noise_models.jl
export S, get_preset_params

"""
    S(ω; a=0, b=0, c=1, d=0, e=0, f=0, ω_c=100, gaussians=[])

General unified spectral density function S(ω).
Returns a vector of S values over given frequency vector ω.
"""
function S(ω::AbstractVector; a=0.0, b=0.0, α=1.0, c=1.0, d=0.0, e=0.0, f=0.0, ω_c=100.0, gaussians=[])
    ε = 1e-6
    base = a ./ (ω.^2 .+ ε) .+ b ./ ((ω .+ ε).^α) .+ c .+ d .* ω .+ e .* ω.^2 .+ f .* ω.^3
    if ω_c < Inf && ω_c > 0
        base .*= exp.(-ω ./ ω_c)
    end
    # Add Gaussian peaks
    # gaussians is a list of tuples (A, μ, σ)
    gauss_part = zeros(length(ω))
    for g in gaussians
        A, μ, σ = g
        gauss_part .+= A .* exp.(-((ω .- μ).^2) ./ (2 * σ^2))
    end
    
    Total = base .+ gauss_part
    Total[findall(abs.(ω) .< 1e-10)] .= 0.0 # S(0) is manually set to 0 for all omega close to zero.

    return Total
end


"""
    get_preset_params(name::String)

Return a dictionary of parameter sets for standard named noise models.
"""
function get_preset_params(name::String)
    name = lowercase(name)
    if name == "white"
        return Dict(:a => 0, :b => 0, :c => 1, :ω_c => Inf)
    elseif name == "1/f"
        return Dict(:a => 0, :b => 1, :α => 1.0, :c => 0.01, :ω_c => Inf)
    elseif name == "ou"
        return Dict(:a => 1, :c => 0, :ω_c => Inf)
    elseif name == "quasi_static"
        return Dict(:a => 1, :b => 0.1, :c => 0)
    elseif name == "composite"
        return Dict(:a => 0.5, :b => 0.5, :c => 0.2, :e => 0.01)
    elseif name == "white+peak"
        return Dict(:c => 1, :gaussians => [(0.5, 5.0, 0.5)])
    else
        error("Unknown noise model '$name'")
    end
end
