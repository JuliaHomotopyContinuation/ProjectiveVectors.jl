using Documenter, ProjectiveVectors
using LinearAlgebra

makedocs(
    format = :html,
    sitename = "ProjectiveVectors",
    pages = [
        "Index" => "index.html"
        ],
    doctest=false,
    modules=[ProjectiveVectors],
    checkdocs=:exports
)

deploydocs(
    repo   = "github.com/JuliaHomotopyContinuation/ProjectiveVectors.jl.git",
    target = "build",
    julia = "1.0",
    osname = "linux",
    deps   = nothing,
    make   = nothing
)