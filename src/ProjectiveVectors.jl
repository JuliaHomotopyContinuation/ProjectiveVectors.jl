module ProjectiveVectors

import LinearAlgebra

export PVector, data, dims, embed


"""
    AbstractProjectiveVector{T, N} <: AbstractVector{T}

An abstract type representing a vector in a product of `N` projective spaces ``P(T^{dᵢ})``.
"""
abstract type AbstractProjectiveVector{T, N} <: AbstractVector{T} end


"""
    PVector{T, N} <: AbstractProjectiveVector{T, N}

A `PVector` represents a projective vector `z` which lives in a product of `N`
projective spaces ``P(T)^{dᵢ}``. The underlying data structure is a `Vector{T}`.
"""
struct PVector{T, N} <: AbstractProjectiveVector{T, N}
    data::Vector{T}
    dims::NTuple{N, Int} # Projective dimensions

    function PVector{T, N}(data, dims) where {T, N}
        @assert length(data) == sum(dims) + N
        new(data, dims)
    end
end

PVector(z::Vector{T}, dims::NTuple{N, Int}) where {T, N} = PVector{T, N}(z, dims)

PVector(vectors::Vector...) = PVector(promote(vectors...))
function PVector(vectors::NTuple{N, Vector{T}}) where {T, N}
    data = reduce(vcat, vectors)
    dims = _dim.(vectors)
    PVector(data, dims)
end
_dim(x::Vector) = length(x) - 1



"""
    data(z::AbstractProjectiveVector)

Access the underlying vector of `z`. This is useful to pass
the vector into some function which does not know the
projective structure.

    data(z::AbstractVector)

For general `AbstractVector`s this is just the identity.
"""
data(z::PVector) = z.data
data(z::AbstractVector) = z


"""
    dims(z::PVector)

Dimensions of the projective spaces in which `z` lives.
"""
dims(z::PVector) = z.dims

##################
# Base overloads
#################

# AbstractArray interface

Base.@propagate_inbounds Base.getindex(z::PVector, k) = getindex(z.data, k)
Base.@propagate_inbounds Base.setindex!(z::PVector, zᵢ) = setindex!(z.data, zᵢ)
Base.firstindex(z::PVector) = 1
Base.lastindex(z::PVector) = length(z)

Base.length(z::PVector) = length(z.data)
Base.size(z::PVector) = (length(z),)

Base.@propagate_inbounds function Base.getindex(z::PVector, i::Integer, j::Integer)
    d = dims(z)
    @boundscheck checkbounds(z, i, j)
    k = 0
    for l = 1:i-1
        k += d[l] + 1
    end
    z[k + j]
end
function Base.checkbounds(z::PVector{T, N}, i, j) where {T, N}
    if i < 1 || i > N
        error("Attempt to access product of $N projective spaces at index $i")
    end
    dᵢ = dims(z)[i]
    if j < 1 || j > dᵢ + 1
        error("Attempt to access $(dᵢ)-dimensional projective space at index $i")
    end
    true
end

# conversion
Base.similar(v::PVector, ::Type{T}) where T = PVector(similar(v.data, T), v.dims)
function Base.convert(::Type{PVector{T, N}}, z::PVector{T1, N}) where {T, N, T1}
    PVector(convert(Vector{T}, z.data), z.dims)
end

Base.show(io::IO, ::MIME"text/plain", z::PVector) = show(io, z)
function Base.show(io::IO, z::PVector{T, N}) where {T, N}
    if !(get(io, :compact, false))
        print(io, "PVector{$T, $N}:\n ")
    end
    for (i, dᵢ) in enumerate(dims(z))
        if i > 1
            print(io, " × ")
        end
        print(io, "[")
        for j=1:(dᵢ + 1)
            print(io, z[i, j])
            if j ≤ dᵢ
                print(io, ", ")
            end
        end
        print(io, "]")
    end
end
Base.show(io::IO, ::MIME"application/juno+inline", z::PVector) = show(io, z)



"""
    embed(x::AbstractVector{T}, dims::NTuple{N, Int})::PVector{T, N}

Embed an affine vector `x` in a product of affine spaces by the map πᵢ: xᵢ -> [xᵢ; 1]
for each subset `xᵢ` of `x` according to `dims`.

## Examples
```julia-repl

```
"""
function embed(z::AbstractVector{T}, dims::NTuple{N, Int}) where {T, N}
    n = sum(dims)
    if length(z) ≠ n
        error("Cannot embed `x` since passed dimensions `dims` are invalid for the given vector `x`.")
    end
    data = Vector{T}(undef, n+N)
    k = 1
    j = 1
    for dᵢ in dims
        for _ = 1:dᵢ
            data[k] = z[j]
            k += 1
            j += 1
        end
        data[k] = one(T)
        k += 1
    end

    PVector(data, dims)
end
function embed(vectors::NTuple{N, Vector{T}}) where {T, N}
    data = reduce(vcat, vectors)
    dims = length.(vectors)
    embed(data, dims)
end
embed(vectors::Vector...) = embed(promote(vectors...))

end
