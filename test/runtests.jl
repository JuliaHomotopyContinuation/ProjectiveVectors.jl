using Test
using ProjectiveVectors

@testset "ProjectiveVectors" begin
    @testset "Constructor" begin
        x = PVector([1, 2, 3])
        @test x isa PVector{Int, 1}
        @test data(x) == [1, 2, 3]
        @test dims(x) == (2,)

        x = PVector([1, 2, 3, 4, 5, 6, 7, 8], (2, 2, 1))
        @test x isa PVector{Int, 3}
        @test dims(x) == (2, 2, 1)
        @test PVector([1, 2, 3], [4, 5, 6], [7, 8]) == x
    end

    @testset "embed" begin
        z = embed([2, 3])
        @test z isa PVector{Int, 1}
        @test data(z) == [2, 3, 1]

        z = embed([2, 3], [4, 5, 6])
        @test z isa PVector{Int, 2}
        @test data(z) == [2, 3, 1, 4, 5, 6, 1]
        @test dims(z) == (2, 3)

        z = embed([2, 3, 4, 5, 6, 7], (2, 3, 1))
        @test z isa PVector{Int, 3}
        @test dims(z) == (2, 3, 1)
        @test data(z) == [2, 3, 1, 4, 5, 6, 1, 7, 1]
    end

    @testset "Show" begin
        z = PVector([2, 3, 4, 5, 6, 7])
        @test sprint(show, z) == "PVector{Int64, 1}:\n [2, 3, 4, 5, 6, 7]"
        @test sprint(show, z, context=:compact => true) == "[2, 3, 4, 5, 6, 7]"

        z = embed([2, 3, 4, 5, 6, 7], (2, 3, 1))
        @test sprint(show, z) == "PVector{Int64, 3}:\n [2, 3, 1] × [4, 5, 6, 1] × [7, 1]"
        @test sprint(show, z, context=:compact => true) == "[2, 3, 1] × [4, 5, 6, 1] × [7, 1]"
    end
end
