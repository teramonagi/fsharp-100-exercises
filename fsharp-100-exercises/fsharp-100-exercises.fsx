(*** hide ***)
#load "packages/FsLab/FsLab.fsx"

(**
<a href="https://github.com/teramonagi/fsharp-100-exercises"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/38ef81f8aca64bb9a64448d0d70f1308ef5341ab/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6461726b626c75655f3132313632312e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_darkblue_121621.png"></a>

# 100 FSharp exercises
This is FSharp version of [100 numpy exercises](http://www.loria.fr/~rougier/teaching/numpy.100/)

Latest version of 100 numpy excercises are available at [this repository](https://github.com/rougier/numpy-100).

The source of this document is available at https://github.com/teramonagi/fsharp-100-exercises .
And your pull request is always welcome!!!

In this document, F# core library and [Math.NET Numerics](http://numerics.mathdotnet.com/) are used.
*)

(**
# Neophyte
## 1. Import the numpy package under the name `np`
We use Math.NET Numerics libraries instead of numpy.
*)
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Statistics
(**
## 2. Print the Fsharp version and the configuration.
*)
(*** define-output:loading ***)
printfn "%A" (System.Reflection.Assembly.GetExecutingAssembly().ImageRuntimeVersion)
(*** include-output:loading ***)

(**
## 3. Create a null vector of size 10
### F# Core Library
*)
(*** define-output:neophyte_3_c ***)
let Z = Array.zeroCreate<float>(10)
printfn "%A" Z
(*** include-output:neophyte_3_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:neophyte_3_m ***)
let Z = DenseVector.zero<float>(10)
printfn "%A" (Z.ToArray())
(*** include-output:neophyte_3_m ***)

(**
## 4. Create a null vector of size 10 but the fifth value which is 1
### F# Core Library
*)
(*** define-output:neophyte_4_c ***)
let Z = Array.zeroCreate<float>(10)
Z.[4] <- 1.0
printfn "%A" Z
(*** include-output:neophyte_4_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:neophyte_4_m ***)
let Z = DenseVector.zero<float>(10)
Z.[4] <- 1.0
printfn "%A" (Z.ToArray())
(*** include-output:neophyte_4_m ***)

(**
## 5. Create a vector with values ranging from 10 to 49
### F# Core Library
*)
(*** define-output:neophyte_5_c ***)
let Z = (Array.init 40 (fun index -> index+10))
printfn "%A" Z
(*** include-output:neophyte_5_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:neophyte_5_m ***)
let Z = DenseVector.range 10 1 49
printfn "%A" (Z.ToArray())
(*** include-output:neophyte_5_m ***)

(**
## 6. Create a 3x3 matrix with values ranging from 0 to 8
### F# Core Library
*)
(*** define-output:neophyte_6_c ***)
let Z = Array2D.init 3 3 (fun i j -> 3*i + j)
printfn "%A" Z
(*** include-output:neophyte_6_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:neophyte_6_m ***)
let Z = DenseMatrix.init 3 3 (fun i j -> float i * 3.0 + float j)
printfn "%A" (Z.ToArray())
(*** include-output:neophyte_6_m ***)

(**
## 7. Find indices of non-zero elements from [1,2,0,0,4,0]
### F# Core Library
*)
(*** define-output:neophyte_7_c_1 ***)
let Z = [|1; 2; 0; 0; 4; 0|] |> Array.mapi (fun i x -> (x<>0, i)) |> Array.filter fst |> Array.map snd
printfn "%A" Z
(*** include-output:neophyte_7_c_1 ***)
(** ...Or, you can write as the following:*)
(*** define-output:neophyte_7_c_2 ***)
let Z = [|1; 2; 0; 0; 4; 0|] |> Array.mapi (fun i x -> (x<>0, i)) |> Array.choose (fun x -> if fst x then Some(snd x) else None)
printfn "%A" Z
(*** include-output:neophyte_7_c_2 ***)
(**
### Math.NET Numerics
You can use "Find" method implemented in Vector class directly if you can use [MathNet.Numerics.FSharp 3.6.0](https://www.nuget.org/packages/MathNet.Numerics.FSharp).
(My current MathNet.Numerics(.FSharp) is 3.5.0...)
*)
(*** define-output:neophyte_7_m ***)
let Z = vector [1.; 2.; 0.; 0.; 4.; 0.] |> 
    Vector.foldi (fun i x y -> if y <> 0.0 then Array.append x [|y|] else x) [||]
printfn "%A" Z
(*** include-output:neophyte_7_m ***)

(**
## 8. Create a 3x3 identity matrix
### F# Core Library
*)
(*** define-output:neophyte_8_c ***)
let Z = Array2D.init 3 3 (fun i j -> if i=j then 1 else 0)
printfn "%A" Z
(*** include-output:neophyte_8_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:neophyte_8_m ***)
let Z = DenseMatrix.identity<float> 3
printfn "%A" Z
(*** include-output:neophyte_8_m ***)

(**
## 9. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
### F# Core Library
*)
(*** define-output:neophyte_9_c ***)
let Z = Array2D.init 5 5 (fun i j -> if (i=j+1) then i else 0)
printfn "%A" Z
(*** include-output:neophyte_9_c ***)
(** 
### Math.NET Numerics 
*)
(*** define-output:neophyte_9_m ***)
let Z = DenseMatrix.init 5 5 (fun i j -> if (i=j+1) then float i else 0.0)
printfn "%A" Z
(*** include-output:neophyte_9_m ***)

(**
## 10. Create a 3x3x3 array with random values
### F# Core Library
*)
(*** define-output:neophyte_10-1_c ***)
let rand = System.Random()
let Z = Array3D.init<float> 3 3 3 (fun _ _ _ -> rand.NextDouble())
for z in [0..2] do
    printfn "component: %d" z
    for y in [0..2] do
        for x in [0..2] do
            printf "%f" Z.[x, y, z]
        printfn ""
(*** include-output:neophyte_10-1_c ***)
(**
...or, since there is no strong support for Array3D class in F#, you may write the following :
*)
(*** define-output:neophyte_10-2_c ***)
let rand = System.Random()
let Z = Array.init 3 (fun z -> Array2D.init<float> 3 3 (fun _ _ -> rand.NextDouble()))
for z in [0..2] do
    printfn "component: %d" z
    printfn "%A" Z.[z]
(*** include-output:neophyte_10-2_c ***)
(** 
### Math.NET Numerics 
*)
(*** define-output:neophyte_10_m ***)
//I used the type "Array of DenseMatix" here because there is no 3D Vector in core Math.NET.
let rand = new MathNet.Numerics.Distributions.ContinuousUniform()
let Z = Array.init 3 (fun _ -> DenseMatrix.random<float> 3 3 rand)
for z in [0..2] do
    printfn "component: %d" z
    printfn "%A" Z.[z]
(*** include-output:neophyte_10_m ***)


(**
# Novice
## 1. Create a 8x8 matrix and fill it with a checkerboard pattern
### F# Core Library
*)
(*** define-output:novice_1_c ***)
let Z = Array2D.init 8 8 (fun i j -> (i+j)%2)
printfn "%A" Z
(*** include-output:novice_1_c ***)
(** 
### Math.NET Numerics 
*)
(*** define-output:neophyte_1_m ***)
let Z = DenseMatrix.init 8 8 (fun i j -> float ((i+j)%2))
printfn "%A" Z
(*** include-output:neophyte_1_m ***)

(**
## 2. Create a 10x10 array with random values and find the minimum and maximum values
### F# Core Library
*)
(*** define-output:novice_2_c ***)
let rand = System.Random()
let Z = Array2D.init 10 10 (fun i j -> rand.NextDouble())
let Zmin = Z |> Seq.cast<float> |> Seq.min
let Zmax = Z |> Seq.cast<float> |> Seq.max
printfn "%f, %f" Zmin Zmax
(*** include-output:novice_2_c ***)
(** 
### Math.NET Numerics 
*)
(*** define-output:neophyte_2_m ***)
let rand = new MathNet.Numerics.Distributions.ContinuousUniform()
let Z = DenseMatrix.random<float> 10 10 rand
let Zmin = Matrix.reduce min Z  
let Zmax = Matrix.reduce max Z  
printfn "%f, %f" Zmin Zmax
(*** include-output:neophyte_2_m ***)

(**
## 3. Create a checkerboard 8x8 matrix using the tile function
There is no numpy's tile equivalent function in F# and Math.NET. These are another solutions for Novice 1.
### F# Core Library
*)
(*** define-output:novice_3_c ***)
let Z = Array.init 8 (fun i -> Array.init 8 (fun j -> (i+j)%2)) |> array2D
printfn "%A" Z
(*** include-output:novice_3_c ***)
(** 
### Math.NET Numerics 
*)
(*** define-output:neophyte_3_m ***)
let seq01 = Seq.initInfinite (fun i -> float(i%2))
let Z = DenseMatrix.initColumns 8 (fun i -> seq01 |> Seq.skip i |> Seq.take 8 |> DenseVector.ofSeq)
printfn "%A" Z
(*** include-output:neophyte_3_m ***)

(**
## 4. Normalize a 5x5 random matrix (between 0 and 1)
### F# Core Library
*)
(*** define-output:novice_4_c ***)
let rand = System.Random()
let Z = Array2D.init 5 5 (fun i j -> rand.NextDouble())
let Zmin = Z |> Seq.cast<float> |> Seq.min
let Zmax = Z |> Seq.cast<float> |> Seq.max
let Z2 = Z |> Array2D.map (fun z -> (z-Zmin)/(Zmax-Zmin))
printfn "%A" Z2
(*** include-output:novice_4_c ***)

(** 
### Math.NET Numerics 
*)
(*** define-output:novice_4_m ***)
let rand = new MathNet.Numerics.Distributions.ContinuousUniform()
let Z = DenseMatrix.random<float> 5 5 rand
let Zmin = Matrix.reduce min Z  
let Zmax = Matrix.reduce max Z  
let Z2 = Matrix.map (fun z -> (z-Zmin)/(Zmax-Zmin) ) Z
printfn "%A" Z2
(*** include-output:novice_4_m ***)

(**
## 5. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
### F# Core Library
*)
(*** define-output:novice_5_c ***)
let x = Array2D.create 5 3 1.0
let y = Array2D.create 3 2 1.0
let inner_product x y = Array.fold2 (fun s x y -> s+x*y) 0.0 x y
let Z = Array2D.init 5 2 (fun i j -> inner_product x.[i,*] y.[*,j] )
printfn "%A" Z
(*** include-output:novice_5_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:novice_5_m ***)
let Z = (DenseMatrix.create 5 3 1.0) * (DenseMatrix.create 3 2 1.0)
printfn "%A" Z
(*** include-output:novice_5_m ***)


(**
## 6. Create a 5x5 matrix with row values ranging from 0 to 4
### F# Core Library
*)
(*** define-output:novice_6_c ***)
let Z = Array2D.init 5 5 (fun i j -> j)
printfn "%A" Z
(*** include-output:novice_6_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:novice_6_m ***)
let Z = DenseMatrix.init 5 5 (fun i j -> float j)
printfn "%A" Z
(*** include-output:novice_6_m ***)

(**
## 7. Create a vector of size 10 with values ranging from 0 to 1, both excluded
It it little bit different from numpy's answer because there is no equivalent method with numpy's linspace in F#/Math.NET.
### F# Core Library
*)
(*** define-output:novice_7_c ***)
let Z = Array.init 12 (fun i -> 1.0/11.0*(float i)) 
let Z2 = Array.sub Z 1 10
printfn "%A" Z2
(*** include-output:novice_7_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:novice_7_m ***)
let Z = (DenseVector.rangef 0.0 (1.0/11.0) 1.0).SubVector(1, 10)
printfn "%A" Z
(*** include-output:novice_7_m ***)

(**
## 8. Create a random vector of size 10 and sort it
### F# Core Library
*)
(*** define-output:novice_8_c ***)
let rand = System.Random()
let Z = Array.init 10 (fun _ -> rand.NextDouble()) |> Array.sort 
printfn "%A" Z
(*** include-output:novice_8_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:novice_8_m ***)
let rand = new MathNet.Numerics.Distributions.ContinuousUniform()
let Z = DenseVector.random<float> 10 rand
MathNet.Numerics.Sorting.Sort(Z)
printfn "%A" Z
(*** include-output:novice_8_m ***)


(**
## 9. Consider two random array A anb B, check if they are equal.
### F# Core Library
*)
(*** define-output:novice_9_c ***)
let rand = System.Random()
let A = Array.init 5 (fun _ -> rand.Next(2))
let B = Array.init 5 (fun _ -> rand.Next(2))
let equal = Array.forall2 (=) A B
printfn "%A" equal
(*** include-output:novice_9_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:novice_9_m ***)
let rand = new MathNet.Numerics.Distributions.DiscreteUniform(0, 1)
let A = rand.Samples() |> Seq.take 5 |> Seq.map float |> DenseVector.ofSeq
let B = rand.Samples() |> Seq.take 5 |> Seq.map float |> DenseVector.ofSeq
let equal = A=B
printfn "%A" equal
(*** include-output:novice_9_m ***)

(**
## 10. Create a random vector of size 30 and find the mean value
### F# Core Library
*)
(*** define-output:novice_10_c ***)
let rand = System.Random()
let Z = Array.init 30 (fun _ -> rand.NextDouble())
let m = Z |> Array.average
printfn "%f" m
(*** include-output:novice_10_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:novice_10_m ***)
let rand = new MathNet.Numerics.Distributions.ContinuousUniform()
let Z = DenseVector.random<float> 30 rand
let m = Z |> Statistics.Mean
printfn "%f" m
(*** include-output:novice_10_m ***)

(**
# Apprentice
## 1. Make an array immutable (read-only)
### F# Core Library
*)
//List is immutable in F# but it does not allow us to do random access.
let Z = List.init 10 (fun _ -> 0)
//It does not work.
//Z.[0] <- 1
(**
### Math.NET Numerics
*)
//There is no way to make an array immutable.

(**
## 2. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
### F# Core Library
*)
(*** define-output:apprentice_2_c ***)
let rand = System.Random()
let Z = Array2D.init 10 2 (fun _ _ -> rand.NextDouble())
let X, Y = Z.[*,0], Z.[*,1]
let R = Array.map2 (fun x y -> sqrt(x*x+y*y)) X Y
let T = Array.map2 (fun x y -> System.Math.Atan2(y, x)) X Y
printfn "%A" R
printfn "%A" T
(*** include-output:apprentice_2_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:apprentice_2_m ***)
let rand = new MathNet.Numerics.Distributions.ContinuousUniform()
let Z = DenseMatrix.random<float> 10 2 rand
let X, Y = Z.[*,0], Z.[*,1]
let R = X.*X + Y.*Y |> Vector.map sqrt
let T = Y./X |> Vector.map System.Math.Atan
printfn "%A" R
printfn "%A" T
(*** include-output:apprentice_2_m ***)

(**
## 3. Create random vector of size 10 and replace the maximum value by 0
### F# Core Library
*)
(*** define-output:apprentice_3_c ***)
let rand = System.Random()
let Z = Array.init 10 (fun _ -> rand.NextDouble())
let Zmax = Array.max Z
Z.[Array.findIndex ((=) Zmax) Z] <- 0.
printfn "%A" Z
(*** include-output:apprentice_3_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:apprentice_3_m ***)
let rand = MathNet.Numerics.Distributions.ContinuousUniform()
let Z = DenseVector.random<float> 10 rand
Z.[Vector.maxIndex Z] <- 0.
printfn "%A" Z
(*** include-output:apprentice_3_m ***)

(**
## 4. Create a structured array with ``x`` and ``y`` coordinates covering the [0,1]x[0,1] area.
There is no way to assign name to Array2D/Matrix. 
We might should use [Deedle](http://bluemountaincapital.github.io/Deedle/) in this situation.
### F# Core Library
*)
(*** define-output:apprentice_4_c ***)
let element = Array.init 10 (fun i -> 1.0/9.0*(float i)) 
let Z = Array2D.init 10 10 (fun i j -> (element.[j], element.[i]))
printfn "%A" Z
(*** include-output:apprentice_4_c ***)
(**
### Math.NET Numerics
*)
(*** define-output:apprentice_4_m ***)
//We can not use the type Matrix<(float, float)> type as a output.
let element = DenseVector.rangef 0.0 (1.0/9.0) 1.0
let Z = Array2D.init 10 10 (fun i j -> (element.[j], element.[i]))
printfn "%A" Z
(*** include-output:apprentice_4_m ***)


(**
## 5. Print the minimum and maximum representable value for each numpy scalar type
These are not types but functions for type conversion like int8, float32 in the below codes.
I used these functions to get its type with Reflection.
*)
(*** define-output:apprentice_5_1 ***)
[int8.GetType(); int32.GetType(); int64.GetType()] |> 
    List.iter (fun x -> 
        printfn "%A" (x.BaseType.GenericTypeArguments.[1].GetField("MinValue").GetValue())
        printfn "%A" (x.BaseType.GenericTypeArguments.[1].GetField("MaxValue").GetValue()))        
(*** include-output:apprentice_5_1 ***)

(*** define-output:apprentice_5_2 ***)
[float32.GetType(); float.GetType()] |> 
    List.iter (fun x -> 
        printfn "%A" (x.BaseType.GenericTypeArguments.[1].GetField("MinValue").GetValue())
        printfn "%A" (x.BaseType.GenericTypeArguments.[1].GetField("MaxValue").GetValue())        
        printfn "%A" (x.BaseType.GenericTypeArguments.[1].GetField("Epsilon").GetValue()))
(*** include-output:apprentice_5_2 ***)

(**
... To be continued.
*)