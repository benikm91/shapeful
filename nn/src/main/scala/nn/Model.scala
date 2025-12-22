package nn

import shapeful.*

object Sequential:
    def |>>[In <: Tuple : Labels](fs: List[Function[Tensor[In], Tensor[In]]]) = 
        fs.foldLeft(identity[Tensor[In]])((acc, g) => acc.andThen(g))
    def |> [In <: Tuple, Out <: Tuple: Labels](f: Tensor[In] => Tensor[Out]) = 
        Sequential(f)

case class Sequential[In <: Tuple, Out <: Tuple: Labels](f: Tensor[In] => Tensor[Out]) extends (Tensor[In] => Tensor[Out]):
    def |>>(gs: List[Function[Tensor[Out], Tensor[Out]]]): Sequential[In, Out] = gs.foldLeft(this)((acc, g) => acc.andThen(g))
    def |> [T2 <: Tuple: Labels](g: Function[Tensor[Out], Tensor[T2]]): Sequential[In, T2] = andThen(g)
    def andThen[T2 <: Tuple: Labels](g: Function[Tensor[Out], Tensor[T2]]): Sequential[In, T2] = Sequential(f.andThen(g))
    def apply(input: Tensor[In]): Tensor[Out] = f(input)

object Residual:
    def |> [In <: Tuple, Out <: Tuple : Labels](f: Tensor[In] => Tensor[Out]) = 
        Residual(f)
    def toResidual[T <: Tuple : Labels](f: Tensor[T] => Tensor[T]): Tensor[T] => Tensor[T] = 
        (t: Tensor[T]) => t + f(t)
    
case class Residual[In <: Tuple, Out <: Tuple : Labels](f: Tensor[In] => Tensor[Out]) extends (Tensor[In] => Tensor[Out]):
    def |> (g: Tensor[Out] => Tensor[Out]): Residual[In, Out] = andThen(Residual.toResidual(g))
    def andThen(g: Tensor[Out] => Tensor[Out]): Residual[In, Out] = Residual(f.andThen(Residual.toResidual(g)))
    def apply(input: Tensor[In]): Tensor[Out] = f(input)