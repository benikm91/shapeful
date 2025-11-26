package shapeful.nn

import shapeful.*
import shapeful.autodiff.{TensorTree, ToPyTree}
import shapeful.nn.Layer1D
import shapeful.nn.Linear.xavier
import shapeful.random.Random.Key
import scala.CanEqual.derived
import scala.deriving.Mirror
import scala.compiletime.{erasedValue, summonInline}

class Linear[In <: Label, Out <: Label]:

  def apply(params: Linear.Params[In, Out])(input: Tensor1[In]): Tensor1[Out] =
    input.matmul(params.weight) + params.bias

object Linear:
  case class Params[InDim <: Label, OutDim <: Label](
      weight: Tensor2[InDim, OutDim],
      bias: Tensor1[OutDim]
  ) derives TensorTree,
        ToPyTree

  // Convenience method for common initialization
  def xavier[InDim <: Label, OutDim <: Label](
      key: Key
  )(using inDim: Dim[InDim], outDim: Dim[OutDim]): Linear.Params[InDim, OutDim] =
    val scale = math.sqrt(2.0 / (inDim.dim + outDim.dim)).toFloat
    Params(
      weight = Tensor.randn(key, Shape(Axis[InDim] -> inDim.dim, Axis[OutDim] -> outDim.dim)) * Tensor0(scale),
      bias = Tensor.zeros(Shape(Axis[OutDim] -> outDim.dim))
    )

  // He initialization for ReLU activation functions
  def he[InDim <: Label, OutDim <: Label](
      key: Key
  )(using inDim: Dim[InDim], outDim: Dim[OutDim]): Linear.Params[InDim, OutDim] =
    val scale = math.sqrt(2.0 / inDim.dim).toFloat
    Params(
      weight = Tensor.randn(key, Shape(Axis[InDim] -> inDim.dim, Axis[OutDim] -> outDim.dim)) * Tensor0(scale),
      bias = Tensor.zeros(Shape(Axis[OutDim] -> outDim.dim))
    )

  given xavierInit[In <: Label, Out <: Label](using inDim: Dim[In], outDim: Dim[Out]): Init[Linear.Params[In, Out]] = 
    new Init[Linear.Params[In, Out]]:
      def apply(key: Key): Linear.Params[In, Out] = Linear.xavier[In, Out](key)


trait Init[T]:
  def apply(key: Key): T

object Init:

  def init[T](key: Key)(using init: Init[T]): T = init.apply(key)

  given Init[EmptyTuple] with
    def apply(key: Key): EmptyTuple = EmptyTuple

  given [H, T <: Tuple](using head: Init[H], tail: Init[T]): Init[H *: T] with
    def apply(key: Key): H *: T =
      val (k1, k2) = key.split2()
      head.apply(k1) *: tail.apply(k2)

  inline given derived[T](using m: Mirror.ProductOf[T]): Init[T] =

    val elemInit = summonInline[Init[m.MirroredElemTypes]]
    
    new Init[T]:
      def apply(key: Key): T = 
        val tuple = elemInit(key)
        m.fromProduct(tuple)