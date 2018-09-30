var documenterSearchIndex = {"docs": [

{
    "location": "index.html#ProjectiveVectors.PVector",
    "page": "Index",
    "title": "ProjectiveVectors.PVector",
    "category": "type",
    "text": "PVector{T, N} <: AbstractProjectiveVector{T, N}\n\nA PVector represents a projective vector z which lives in a product of N projective spaces P(T)^dᵢ. The underlying data structure is a Vector{T}.\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.data",
    "page": "Index",
    "title": "ProjectiveVectors.data",
    "category": "function",
    "text": "data(z::AbstractProjectiveVector)\n\nAccess the underlying vector of z. This is useful to pass the vector into some function which does not know the projective structure.\n\ndata(z::AbstractVector)\n\nFor general AbstractVectors this is just the identity.\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.dims",
    "page": "Index",
    "title": "ProjectiveVectors.dims",
    "category": "function",
    "text": "dims(z::PVector)\n\nDimensions of the projective spaces in which z lives.\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.embed",
    "page": "Index",
    "title": "ProjectiveVectors.embed",
    "category": "function",
    "text": "embed(xs::Vector...; normalize=false)\nembed(x::AbstractVector{T}, dims::NTuple{N, Int}; normalize=false)::PVector{T, N}\n\nEmbed an affine vector x in a product of affine spaces by the map πᵢ: xᵢ -> [xᵢ; 1] for each subset xᵢ of x according to dims. If normalize is true the vector is normalized.\n\nExamples\n\njulia> embed([2, 3])\nPVector{Int64, 1}:\n [2, 3, 1]\n\njulia> embed([2, 3], [4, 5, 6])\nPVector{Int64, 2}:\n [2, 3, 1] × [4, 5, 6, 1]\n\njulia> embed([2.0, 3, 4, 5, 6, 7], (2, 3, 1))\nPVector{Float64, 3}:\n [2.0, 3.0, 1.0] × [4.0, 5.0, 6.0, 1.0] × [7.0, 1.0]\n\n julia> embed([2.0, 3, 4, 5, 6, 7], (2, 3, 1), normalize=true)\n PVector{Float64, 3}:\n  [0.5345224838248488, 0.8017837257372732, 0.2672612419124244] × [0.45291081365783825, 0.5661385170722978, 0.6793662204867574, 0.11322770341445956] × [0.9899494936611666, 0.1414213562373095]\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.dimension_indices",
    "page": "Index",
    "title": "ProjectiveVectors.dimension_indices",
    "category": "function",
    "text": "dimension_indices(z::PVector{T, N})\ndimension_indices(dims::NTuple{N, Int})\n\nReturn a tuple of N UnitRanges indexing the underlying data.\n\nExample\n\njulia> v = PVector([4, 5, 6], [2, 3], [1, 2])\nPVector{Int64, 3}:\n [4, 5, 6] × [2, 3] × [1, 2]\n\njulia> dimension_indices(v)\n(1:3, 4:5, 6:7)\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.hom_dimension_indices",
    "page": "Index",
    "title": "ProjectiveVectors.hom_dimension_indices",
    "category": "function",
    "text": "hom_dimension_indices(z::PVector{T, N})\nhom_dimension_indices(dims::NTuple{N, Int})\n\nReturn a tuple of N (UnitRange, Int) tuples indexing the underlying data per vector where the last coordinate in each vector is treated separetely.\n\nExample\n\njulia> v = PVector([4, 5, 6], [2, 3], [1, 2])\nPVector{Int64, 3}:\n [4, 5, 6] × [2, 3] × [1, 2]\n\n julia> hom_dimension_indices(v)\n ((1:2, 3), (4:4, 5), (6:6, 7))\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.affine_chart",
    "page": "Index",
    "title": "ProjectiveVectors.affine_chart",
    "category": "function",
    "text": "affine_chart(z::PVector)\n\nReturn the affine chart corresponding to the projective vector. This can be seen as the inverse of embed.\n\nExample\n\njulia> v = embed([2.0, 3, 4, 5, 6, 7], (2, 3, 1))\nPVector{Float64, 3}:\n [2.0, 3.0, 1.0] × [4.0, 5.0, 6.0, 1.0] × [7.0, 1.0]\n\njulia> affine_chart(v)\n6-element Array{Float64,1}:\n 2.0\n 3.0\n 4.0\n 5.0\n 6.0\n 7.0\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.affine_chart!",
    "page": "Index",
    "title": "ProjectiveVectors.affine_chart!",
    "category": "function",
    "text": "affine_chart!(x, z::PVector)\n\nInplace variant of affine_chart.\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.norm_affine_chart",
    "page": "Index",
    "title": "ProjectiveVectors.norm_affine_chart",
    "category": "function",
    "text": " norm_affine_chart(z::PVector, p::Real=2) where {T, N}\n\nCompute the p-norm of z on it\'s affine_chart.\n\n\n\n\n\n"
},

{
    "location": "index.html#LinearAlgebra.norm",
    "page": "Index",
    "title": "LinearAlgebra.norm",
    "category": "function",
    "text": "LinearAlgebra.norm(z::PVector{T,N}, p::Real=2)::NTuple{N, real(T)}\n\nCompute the p-norm of z per vector space. This always returns a Tuple.\n\nExample\n\njulia> norm(embed([1, 2, 3, 4, 5], (2, 3)))\n(2.449489742783178, 7.14142842854285)\n\njulia> norm(embed([1, 2, 3, 4, 5]))\n(7.483314773547883,)\n\n\n\n\n\n"
},

{
    "location": "index.html#LinearAlgebra.normalize!",
    "page": "Index",
    "title": "LinearAlgebra.normalize!",
    "category": "function",
    "text": "LinearAlgebra.normalize!(z::PVector, p::Real=2)\n\nNormalize each component of z separetly.\n\n\n\n\n\n"
},

{
    "location": "index.html#LinearAlgebra.normalize",
    "page": "Index",
    "title": "LinearAlgebra.normalize",
    "category": "function",
    "text": "LinearAlgebra.normalize(z::PVector{T, N}, p::Real=2)::PVector{T,N}\n\nNormalize each component of z separetly.\n\n\n\n\n\n"
},

{
    "location": "index.html#LinearAlgebra.dot",
    "page": "Index",
    "title": "LinearAlgebra.dot",
    "category": "function",
    "text": "LinearAlgebra.dot(v::PVector{T, N}, w::PVector{T2, N}) where {T, T2, N}\n\nCompute the component wise dot product. If decorated with @inbounds the check of the dims of v and w is skipped.\n\n\n\n\n\n"
},

{
    "location": "index.html#LinearAlgebra.rmul!",
    "page": "Index",
    "title": "LinearAlgebra.rmul!",
    "category": "function",
    "text": "LinearAlgebra.rmul!(z::PVector{T, N}, λ::NTuple{N, <:Number}) where {T, N}\nLinearAlgebra.rmul!(z::PVector{T, 1}, λ::Number) where {T}\n\nMultiply each component of zᵢ of z by λ[i].\n\n\n\n\n\n"
},

{
    "location": "index.html#ProjectiveVectors.fubini_study",
    "page": "Index",
    "title": "ProjectiveVectors.fubini_study",
    "category": "function",
    "text": "fubini_study(v::PVector{<:Number,N}, w::PVector{<:Number, N})\n\nCompute the Fubini-Study distance between v and w for each component vᵢ and wᵢ. This is defined as arccosvᵢwᵢ.\n\n\n\n\n\n"
},

{
    "location": "index.html#",
    "page": "Index",
    "title": "Index",
    "category": "page",
    "text": "Data structure for elements in products of projective spaces.PVector\ndata\ndims\nembed\ndimension_indices\nhom_dimension_indices\naffine_chart\naffine_chart!\nnorm_affine_chart\nnorm\nnormalize!\nnormalize\ndot\nrmul!\nfubini_study"
},

]}
