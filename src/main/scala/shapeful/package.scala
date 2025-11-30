package object shapeful:

  import tensor.*
  import scala.compiletime.ops.string.+

  type Label = Singleton | Tuple

  object StringMath:
    infix type *[A <: String, B <: String] = A + "*" + B
  
  trait Dim[L <: Label]:
    def dim: Int

  object Dim:
    def apply[L <: Label](dim: Int): Dim[L] =
      val outerDim = dim
      new Dim[L]:
        def dim: Int = outerDim

  export tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export tensor.Shape.{Shape0, Shape1, Shape2, Shape3}
  export tensor.Tensor
  export tensor.{Tensor0, Tensor1, Tensor2, Tensor3}
  export tensor.Tensor.{Tensor0, Tensor1, Tensor2, Tensor3}
  export tensor.DType
  export tensor.Axis
  export linalg.Linalg

  // Export all TensorOps extensions
  export tensor.TensorOps.*
