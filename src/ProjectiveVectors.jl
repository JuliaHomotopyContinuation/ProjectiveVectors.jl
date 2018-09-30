module ProjectiveVectors

using LinearAlgebra
import Base: ==

export PVector, data, dims, embed, dimension_indices, hom_dimension_indices,
    affine_chart, affine_chart!


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

Return a tuple of `N` `UnitRange`s indexing the underlying data.

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

"""
    hom_dimension_indices(z::PVector{T, N})
    hom_dimension_indices(dims::NTuple{N, Int})

Return a tuple of `N` `(UnitRange, Int)` tuples indexing the underlying data per vector
where the last coordinate in each vector is treated separetely.

## Example
```julia-repl
julia> v = PVector([4, 5, 6], [2, 3], [1, 2])
PVector{Int64, 3}:
 [4, 5, 6] × [2, 3] × [1, 2]

 julia> hom_dimension_indices(v)
 ((1:2, 3), (4:4, 5), (6:6, 7))
```
"""
hom_dimension_indices(z::PVector) = hom_dimension_indices(dims(z))
hom_dimension_indices(dims::NTuple{1, Int}) = (1:(dims[1] + 1),)
function hom_dimension_indices(dims::NTuple{N, Int}) where {N}
    k = Ref(1) # we need the ref here to make the compiler happy
    @inbounds map(dims) do dᵢ
        curr_k = k[]
        upper = curr_k + dᵢ
        r = (curr_k:(upper - 1))
        k[] += dᵢ + 1
        (r, upper)
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


LinearAlgebra.norm(z::PVector{T, 1}, p::Real=2) where {T} = (LinearAlgebra.norm(z.data, p),)
@generated function LinearAlgebra.norm(z::PVector{T, N}, p::Real=2) where {T, N}
    quote
        r = dimension_indices(z)
        @inbounds $(Expr(:tuple, (:(_norm_range(z, r[$i], p)) for i=1:N)...))
    end
end
@inline function _norm_range(z::PVector{T}, rᵢ, p) where {T<:Complex}
    normᵢ = zero(real(T))
    if p == 2
        @inbounds for k in rᵢ
            normᵢ += abs2(z[k])
        end
    elseif p == Inf
        @inbounds for k in rᵢ
            normᵢ = max(normᵢ, abs2(z[k]))
        end
    else
        error("p=$p not supported.")
    end
    sqrt(normᵢ)
end

@inline function _norm_range(z::PVector{T}, rᵢ, p) where {T}
    normᵢ = zero(T)
    if p == 2
        @inbounds for k in rᵢ
            normᵢ += abs2(z[k])
        end
        normᵢ = sqrt(normᵢ)
    elseif p == Inf
        @inbounds for k in rᵢ
            normᵢ = max(normᵢ, abs(z[k]))
        end
    else
        error("p=$p not supported.")
    end
    normᵢ
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

function LinearAlgebra.normalize!(z::PVector{T, 1}, p::Real=2) where {T}
    normalize!(z.data, p)
    z
end
LinearAlgebra.normalize!(z::PVector, p::Real=2) = rmul!(z, inv.(LinearAlgebra.norm(z, p)))
LinearAlgebra.normalize(z::PVector, p::Real=2) = normalize!(copy(z), p)

"""
    affine_chart(z::PVector)

Return the affine chart corresponding to the projective vector. This can be seen as the
inverse of [`embed`](@ref).

## Example
```julia-repl
julia> v = embed([2.0, 3, 4, 5, 6, 7], (2, 3, 1))
PVector{Float64, 3}:
 [2.0, 3.0, 1.0] × [4.0, 5.0, 6.0, 1.0] × [7.0, 1.0]

julia> affine_chart(v)
6-element Array{Float64,1}:
 2.0
 3.0
 4.0
 5.0
 6.0
 7.0
```
"""
function affine_chart(z::PVector{T}) where {T}
    @inbounds affine_chart!(Vector{T}(undef, sum(dims(z))), z)
end


"""
    affine_chart!(x, z::PVector)

Inplace variant of [`affine_chart`](@ref)
"""
Base.@propagate_inbounds function affine_chart!(x, z::PVector)
    k = 1
    for (rᵢ, hᵢ) in hom_dimension_indices(z)
        @inbounds normalizer = inv(z[hᵢ])
        for i in rᵢ
            x[k] = z[i] * normalizer
            k += 1
        end
    end
    x
end

Base.@propagate_inbounds function affine_chart!(x, z::PVector{T, 1}) where {T}
    n = length(z)
    v = inv(z[n])
    for i=1:n-1
        x[i] = z[i] * v
    end
    x
end




#
# using LinearAlgebra
# import Base: ==
# import ..Utilities: infinity_norm, unsafe_infinity_norm
#
# export AbstractProjectiveVector,
#     PVector,
#     ProdPVector,
#     raw,
#     homvar,
#     affine,
#     affine!,
#     embed,
#     at_infinity,
#     pvectors,
#     infinity_norm,
#     infinity_norm_fast,
#     unsafe_infinity_norm,
#     converteltype
#
# abstract type AbstractProjectiveVector{T, N} <: AbstractVector{T} end
#
#
# """
#     homvar(z::PVector)
#
# Get the index of the homogenous variable.
# """
# homvar(z::PVector) = z.homvar
#
# (==)(v::PVector, w::PVector) = v.homvar == w.homvar && v.data == w.data
#
# """
#     converteltype(v::PVector, T)
# """
# Base.similar(v::PVector, ::Type{T}) where T = PVector(convert.(T, v.data), v.homvar)
#
# """
#     embed(x::Vector, homvar::Union{Nothing, Int})::PVector
#
# Embed a vector `x` into projective space with homogenization variable `homvar`.
#
#     embed(z::PVector, x::Vector)
#
# Embed a vector `x` into projective space the same way `x` was embedded.
# """
# function embed(x::Vector{T}, homvar) where T
#     k = 1
#     data = Vector{T}(undef, length(x) + 1)
#     @inbounds for k in 1:length(data)
#         if k == homvar
#             data[k] = one(T)
#         else
#             i = k < homvar ? k : k - 1
#             data[k] = x[i]
#         end
#     end
#     PVector(data, homvar)
# end
# embed(z::PVector, x::Vector) = embed(x, z.homvar)
#
#
# """
#     affine_normalizer(z::PVector)
#
# Returns a scalar to bring `z` on its affine patch.
# """
# affine_normalizer(z::PVector{T, Int}) where T = inv(z.data[z.homvar])
# """
#     abs2_ffine_normalizer(z::PVector)
#
# Returns the squared absolute value of the normalizer.
# """
# abs2_affine_normalizer(z::PVector{T, Int}) where T = inv(abs2(z.data[z.homvar]))
#
# """
#     affine(z::PVector{T, Int})::Vector
#
# Return the corresponding affine vector with respect to the standard patch.
#
#     affine(z::PVector, i::Int)::Vector
#
# Return the corresponding affine vector with xᵢ=1.
# """
# affine(z::PVector{T, Int}) where {T} = affine(z, z.homvar)
# function affine(z::PVector{T}, i::Int) where T
#     x = Vector{T}(undef, length(z) - 1)
#     normalizer = inv(z.data[i])
#     @inbounds for k in 1:length(z)
#         if k == i
#             continue
#         end
#         j = k < i ? k : k - 1
#         x[j] = z.data[k] * normalizer
#     end
#     x
# end
#
# """
#     affine!(z::PVector{T, Int})
#
# Bring the projective vector on associated affine patch.
# """
# affine!(z::PVector{T, Int}) where T = LinearAlgebra.rmul!(z.data, affine_normalizer(z))
#
# """
#     infinity_norm(z::PVector{T, Int})
#
# Compute the ∞-norm of `z`. If `z` is a complex vector this is more efficient
# than `norm(z, Inf)`.
#
#     infinity_norm(z₁::PVector{T, Int}, z₂::PVector{T, Int})
#
# Compute the ∞-norm of `z₁-z₂` by bringing both vectors first on their respective
# affine patch. This therefore only makes sense if both vectors have the
# same affine patch.
# """
# function infinity_norm(z::PVector{<:Complex, Int})
#     sqrt(maximum(abs2, raw(z)) * abs2_affine_normalizer(z))
# end
#
# function infinity_norm(z₁::PVector{<:Complex, Int}, z₂::PVector{<:Complex, Int})
#     normalizer₁ = affine_normalizer(z₁)
#     normalizer₂ = -affine_normalizer(z₂)
#     @inbounds m = abs2(muladd(z₁[1], normalizer₁, z₂[1] * normalizer₂))
#     n₁, n₂ = length(z₁), length(z₂)
#     if n₁ ≠ n₂
#         return convert(typeof(m), Inf)
#     end
#     @inbounds for k=2:n₁
#         m = max(m, abs2(muladd(z₁[k], normalizer₁, z₂[k] * normalizer₂)))
#     end
#     sqrt(m)
# end
#
# """
#     unsafe_infinity_norm(z₁::PVector, z₂::PVector)
#
# Compute the ∞-norm of `z₁-z₂`. This *does not* bring both vectors on their respective
# affine patch before computing the norm.
# """
# function unsafe_infinity_norm(z₁::PVector, z₂::PVector)
#     @inbounds m = abs2(z₁[1] - z₂[1])
#     n₁, n₂ = length(z₁), length(z₂)
#     if n₁ ≠ n₂
#         return convert(typeof(m), Inf)
#     end
#     @inbounds for k=2:n₁
#         m = max(m, abs2(z₁[k] - z₂[k]))
#     end
#     sqrt(m)
# end
#
# """
#     at_infinity(z::PVector{T, Int}, maxnorm)
#
# Check whether `z` represents a point at infinity.
# We declare `z` at infinity if the infinity norm of
# its affine vector is larger than `maxnorm`.
# """
# function at_infinity(z::PVector{<:Complex, Int}, maxnorm)
#     at_inf = false
#     @inbounds tol = maxnorm * maxnorm * abs2(z.data[z.homvar])
#     @inbounds for k in 1:length(z)
#         if k == z.homvar
#             continue
#         # we avoid the sqrt here by using the squared comparison
#         elseif abs2(z.data[k]) > tol
#             return true
#         end
#     end
#     false
# end
# function at_infinity(z::PVector{<:Real, Int}, maxnorm)
#     at_inf = false
#     @inbounds tol = maxnorm * abs(z.data[z.homvar])
#     @inbounds for k in 1:length(z)
#         if k == z.homvar
#             continue
#         elseif abs(z.data[k]) > tol
#             return true
#         end
#     end
#     false
# end
#
# LinearAlgebra.norm(v::PVector, p::Real=2) = norm(v.data, p)
# function LinearAlgebra.normalize!(v::PVector, p::Real=2)
#     LinearAlgebra.normalize!(v.data, p)
#     v
# end
#
# LinearAlgebra.dot(v::PVector, w::PVector) = LinearAlgebra.dot(v.data, w.data)
#
# const VecView{T} = SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
# #
# # """
# #     ProdPVector(pvectors)
# #
# # Construct a product of `PVector`s. This
# # """
# # struct ProdPVector{T, H<:Union{Nothing, Int}} <: AbstractProjectiveVector{T}
# #     data::Vector{T}
# #     # It is faster to use a tuple instead but this puts
# #     # additional stress on the compiler and makes things
# #     # not inferable, so we do this not for now.
# #     # But we should be able to change this easily later
# #     pvectors::Vector{PVector{T, H, VecView{T}}}
# # end
# #
# # function ProdPVector(vs)
# #     data = copy(vs[1].data);
# #     for k=2:length(vs)
# #         append!(data, vs[k].data)
# #     end
# #     pvectors = _create_pvectors(data, vs)
# #     ProdPVector(data, pvectors)
# # end
# #
# # function (==)(v::ProdPVector, w::ProdPVector)
# #     for (vᵢ, wᵢ) in zip(pvectors(v), pvectors(w))
# #         if vᵢ != wᵢ
# #             return false
# #         end
# #     end
# #     true
# # end
# #
# # function _create_pvectors(data, vs)
# #     pvectors = Vector{PVector{eltype(data), VecView{eltype(data)}}}()
# #     k = 1
# #     for i = 1:length(vs)
# #         n = length(vs[i])
# #         vdata = view(data, k:k+n-1)
# #         push!(pvectors, PVector(vdata, homvar(vs[i])))
# #         k += n
# #     end
# #     pvectors
# # end
# #
# # function Base.copy(z::ProdPVector)
# #     data = copy(z.data)
# #     new_pvectors = _create_pvectors(data, pvectors(z))
# #     ProdPVector(data, new_pvectors)
# # end
# #
# # function Base.similar(v::ProdPVector, ::Type{T}) where T
# #     data = convert.(T, v.data)
# #     ProdPVector(data, _create_pvectors(data, pvectors(v)))
# # end
# #
# # """
# #     pvectors(z::ProdPVector)
# #
# # Return the `PVector`s out of which the product `z` exists.
# # """
# # pvectors(z::ProdPVector) = z.pvectors
# #
# # """
# #     raw(z::ProdPVector)
# #
# # access_input the underlying vector of the product `z`. Note that this
# # is only a single vector. This is useful to pass
# # the vector into some function which does not know the
# # projective structure.
# # """
# # raw(v::ProdPVector) = v.data
# #
# # """
# #     at_infinity(z::ProdPVector, maxnorm)
# #
# # Returns `true` if any vector of the product is at infinity.
# # """
# # function at_infinity(v::ProdPVector, maxnorm)
# #     for vᵢ in pvectors(v)
# #         if at_infinity(vᵢ, maxnorm)
# #             return true
# #         end
# #     end
# #     false
# # end
# #
# # """
# #     affine(z::ProdPVector)
# #
# # For each projective vector of the product return associated affine patch.
# # """
# # affine(z::ProdPVector) = affine.(z.pvectors)
# #
# # """
# #     affine!(z::ProdPVector)
# #
# # Bring each projective vector of the product on the associated affine patch.
# # """
# # function affine!(v::ProdPVector)
# #     for w in pvectors(v)
# #         affine!(w)
# #     end
# #     v
# # end
# #
# # function LinearAlgebra.normalize!(v::ProdPVector, p::Real=2)
# #     for w in pvectors(v)
# #         normalize!(w, p)
# #     end
# #     v
# # end
# #
# # infinity_norm(z::ProdPVector) = maximum(infinity_norm, pvectors(z))
# # function infinity_norm(v::ProdPVector, w::ProdPVector)
# #     p₁ = pvectors(v)
# #     p₂ = pvectors(w)
# #     m = infinity_norm(p₁[1], p₂[1])
# #     for k=2:length(p₁)
# #         m = max(m, infinity_norm(p₁[k], p₂[k]))
# #     end
# #     m
# # end
#
#
# # AbstractVector interface
#
# Base.size(z::AbstractProjectiveVector) = size(z.data)
# Base.length(z::AbstractProjectiveVector) = length(z.data)
# Base.getindex(z::AbstractProjectiveVector, i::Integer) = getindex(z.data, i)
# Base.setindex!(z::AbstractProjectiveVector, zᵢ, i::Integer) = setindex!(z.data, zᵢ, i)
# Base.lastindex(z::AbstractProjectiveVector) = lastindex(z.data)

end
