module ProjectiveVectors

using LinearAlgebra
import Base: ==

export PVector, data, dims, embed, dimension_indices


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

    Base.copy(v::PVector{T, N}) where {T, N} = new{T, N}(copy(v.data), v.dims)
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


"""
    dimension_indices(z::PVector{T, N})
    dimension_indices(dims::NTuple{N, Int})

Return a tuple of `N` `UnitRanges` indexing the underlying data.

## Example
```julia-repl
julia> v = PVector([4, 5, 6], [2, 3], [1, 2])
PVector{Int64, 3}:
 [4, 5, 6] × [2, 3] × [1, 2]

julia> dimension_indices(v)
(1:3, 4:5, 6:7)
```
"""
dimension_indices(z::PVector) = dimension_indices(dims(z))
dimension_indices(dims::NTuple{1, Int}) = (1:(dims[1] + 1),)
function dimension_indices(dims::NTuple{N, Int}) where {N}
    k = Ref(1)
    @inbounds map(dims) do dᵢ
        curr_k = k[]
        r = (curr_k:(curr_k + dᵢ))
        k[] += dᵢ + 1
        r
    end
end

##################
# Base overloads
#################

# AbstractArray interface

Base.@propagate_inbounds Base.getindex(z::PVector, k) = getindex(z.data, k)
Base.@propagate_inbounds Base.setindex!(z::PVector, v, i) = setindex!(z.data, v, i)
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

# equality
(==)(v::PVector, w::PVector) = dims(v) == dims(w) && v.data == w.data


# show

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
    embed(xs::Vector...)
    embed(x::AbstractVector{T}, dims::NTuple{N, Int})::PVector{T, N}

Embed an affine vector `x` in a product of affine spaces by the map πᵢ: xᵢ -> [xᵢ; 1]
for each subset `xᵢ` of `x` according to `dims`.

## Examples
```julia-repl
julia> p = embed([2, 3])
PVector{Int64, 1}:
 [2, 3, 1]

julia> z = embed([2, 3], [4, 5, 6])
PVector{Int64, 2}:
 [2, 3, 1] × [4, 5, 6, 1]

julia> z = embed([2, 3, 4, 5, 6, 7], (2, 3, 1))
PVector{Int64, 3}:
 [2, 3, 1] × [4, 5, 6, 1] × [7, 1]
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


LinearAlgebra.norm(z::PVector{T, 1}) where {T} = (LinearAlgebra.norm(z.data),)
@generated function LinearAlgebra.norm(z::PVector{T, N}) where {T, N}
    quote
        r = dimension_indices(z)
        @inbounds $(Expr(:tuple, (:(_norm_range(z, r[$i])) for i=1:N)...))
    end
end
@inline function _norm_range(z::PVector{T}, rᵢ) where {T}
    normᵢ = zero(T)
    @inbounds for k in rᵢ
        normᵢ += abs2(z[k])
    end
    sqrt(normᵢ)
end


function LinearAlgebra.rmul!(z::PVector{T, 1}, λ::Number) where {T}
    rmul!(z.data, λ)
    z
end
function LinearAlgebra.rmul!(z::PVector{T, N}, λ::NTuple{N, <:Number}) where {T, N}
    r = dimension_indices(z)
    @inbounds for i = 1:N
        rᵢ, λᵢ = r[i], λ[i]
        for k in rᵢ
            z[k] *= λᵢ
        end
    end
    z
end

function LinearAlgebra.normalize!(z::PVector{T, 1}) where {T}
    normalize!(z.data)
    z
end
LinearAlgebra.normalize!(z::PVector) = rmul!(z, inv.(LinearAlgebra.norm(z)))
end
