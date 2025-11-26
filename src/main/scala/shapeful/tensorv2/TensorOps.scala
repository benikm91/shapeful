package shapeful.tensorv2

import shapeful.Label
import shapeful.jaxv2.Jax
import scala.annotation.targetName
import scala.util.NotGiven
import Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.jax.Jax.PyDynamic
import TupleHelpers.{NamesOf, NamesOfImpl}

object TensorOps:

  extension (l: List[String])
    def removeAt(index: Int): List[String] = l.patch(index, Nil, 1)

  extension [T <: Tuple : NamesOf](t: Tensor[T])

    def vmap[VmapAxis <: Label : ValueOf, OuterShape <: Tuple : NamesOf](
      axis: Axis[VmapAxis]
    )(
        f: Tensor[TupleHelpers.Remove[VmapAxis, T]] => Tensor[OuterShape]
    )(
      using 
      vmapAxisIndex: AxisIndex[VmapAxis, T],
    ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =
      import NamesOf.ForRemove.given
      val fpy = (jxpr: Jax.PyDynamic) =>
        val innerTensor = Tensor[TupleHelpers.Remove[VmapAxis, T]](jxpr)
        val result = f(innerTensor)
        result.jaxValue
      Tensor(Jax.jax_helper.vmap(fpy, vmapAxisIndex.value)(t.jaxValue))


    def applyUniaryJaxF(f: (Jax.PyDynamic => Jax.PyDynamic) | Jax.PyDynamic): Tensor[T] = 
      val ff = f.asInstanceOf[Jax.PyDynamic] // hack to satisfy scala compiler, essentially Jax.PyDynamic => Jax.PyDynamic and Jax.PyDynamic are interchangeable
      t.copy(jaxValue = ff(t.jaxValue))

    def applyBinaryJaxF(other: Tensor[T], f: Jax.PyDynamic): Tensor[T] = 
      t.copy(jaxValue = f(t.jaxValue, other.jaxValue))

    def applyBinaryJaxF2(other: Tensor0, f: Jax.PyDynamic): Tensor[T] = 
      t.copy(jaxValue = f(t.jaxValue, other.jaxValue))

    def +(other: Tensor[T]): Tensor[T] =
      applyBinaryJaxF(other, Jax.jnp.add)

    @targetName("tensor0PlusScalar")
    def +(other: Tensor0): Tensor[T] =
     applyBinaryJaxF2(other, Jax.jnp.add)

    // Subtraction with same shape - most common case
    def -(other: Tensor[T]): Tensor[T] =
      applyBinaryJaxF(other, Jax.jnp.subtract)

    @targetName("tensor0MinusScalar")
    def -(other: Tensor0): Tensor[T] =
      applyBinaryJaxF2(other, Jax.jnp.subtract)

    def *(other: Tensor[T]): Tensor[T] =
      applyBinaryJaxF(other, Jax.jnp.multiply)

    @targetName("tensor0MultScalar")
    def *(other: Tensor0): Tensor[T] =
      applyBinaryJaxF2(other, Jax.jnp.multiply)
    
    @targetName("tensorDivTensor")
    def /(other: Tensor[T]): Tensor[T] =
      applyBinaryJaxF(other, Jax.jnp.divide)

    @targetName("tensorDivScalar")
    def /(other: Tensor0): Tensor[T] =
      applyBinaryJaxF2(other, Jax.jnp.divide)

    def exp: Tensor[T] =
      Tensor(Jax.jnp.exp(t.jaxValue))

    def log: Tensor[T] =
      Tensor(Jax.jnp.log(t.jaxValue))

    def pow(n: Tensor0): Tensor[T] =
      applyBinaryJaxF2(n, Jax.jnp.pow)

    def norm: Tensor0 = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))

    // Reduction operations
    def sum: Tensor0 = Tensor0(Jax.jnp.sum(t.jaxValue))

    def mean: Tensor0 = Tensor0(Jax.jnp.mean(t.jaxValue))

    def min: Tensor0 = Tensor0(Jax.jnp.min(t.jaxValue))

    def max: Tensor0 = Tensor0(Jax.jnp.max(t.jaxValue))

    def argmin: Tensor0 = Tensor0(Jax.jnp.argmin(t.jaxValue))

    def argmax: Tensor0 = Tensor0(Jax.jnp.argmax(t.jaxValue))

    def std: Tensor0 = Tensor0(Jax.jnp.std(t.jaxValue))

    def variance: Tensor0 = Tensor0(Jax.jnp.`var`(t.jaxValue))

    def sum[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.sum(t.jaxValue, axis = axisIndex.value))

    def mean[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.mean(t.jaxValue, axis = axisIndex.value))

    def max[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.max(t.jaxValue, axis = axisIndex.value))

    def min[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.min(t.jaxValue, axis = axisIndex.value))

    def argmax[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.argmax(t.jaxValue, axis = axisIndex.value))

    def argmin[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.argmin(t.jaxValue, axis = axisIndex.value))

    def std[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.std(t.jaxValue, axis = axisIndex.value))

    def variance[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import NamesOf.ForRemove.given
      Tensor(Jax.jnp.`var`(t.jaxValue, axis = axisIndex.value))

    def abs: Tensor[T] = applyUniaryJaxF(Jax.jnp.abs)

    def sign: Tensor[T] = applyUniaryJaxF(Jax.jnp.sign)

    def sqrt: Tensor[T] = applyUniaryJaxF(Jax.jnp.sqrt)

    def sin: Tensor[T] = applyUniaryJaxF(Jax.jnp.sin)

    def cos: Tensor[T] = applyUniaryJaxF(Jax.jnp.cos)

    def tanh: Tensor[T] = applyUniaryJaxF(Jax.jnp.tanh)

    def sigmoid: Tensor[T] =
      val ones = Tensor.ones(t.shape)
      val minust = t * Tensor0(-1.0f)
      ones / (ones + (minust).exp)

    def relu: Tensor[T] = Tensor(Jax.jnp.maximum(t.jaxValue, Jax.jnp.zeros(t.jaxValue.shape)))

    def softmax[SoftmaxAxis <: Label](axis: Axis[SoftmaxAxis])()(using axisIndex: AxisIndex[SoftmaxAxis, T]): Tensor[T] =
      applyUniaryJaxF(Jax.jnn.softmax(_, axis = axisIndex.value))

    def clamp(min: Float, max: Float): Tensor[T] = applyUniaryJaxF(Jax.jnp.clip(_, min, max))

    def clamp(min: Tensor0, max: Tensor0): Tensor[T] = applyUniaryJaxF(Jax.jnp.clip(_, min.jaxValue, max.jaxValue))

    def <(other: Tensor[T]): Tensor[T] = applyUniaryJaxF(Jax.jnp.less(_, other.jaxValue))

    def <=(other: Tensor[T]): Tensor[T] = applyUniaryJaxF(Jax.jnp.less_equal(_, other.jaxValue))

    def >(other: Tensor[T]): Tensor[T] = applyUniaryJaxF(Jax.jnp.greater(_, other.jaxValue))

    def >=(other: Tensor[T]): Tensor[T] = applyUniaryJaxF(Jax.jnp.greater_equal(_, other.jaxValue))

  extension (t: Tensor0)
    def toInt: Int = t.jaxValue.item().as[Int]
    def toFloat: Float = t.jaxValue.item().as[Float]
    def toBool: Boolean = t.jaxValue.item().as[Boolean]

    // def /(other: Tensor0): Tensor0 = Tensor0(Jax.jnp.divide(t.jaxValue, other.jaxValue))

    @targetName("tensor0Pow")
    def pow(exponent: Tensor0): Tensor0 = Tensor0(Jax.jnp.pow(t.jaxValue, exponent.jaxValue))

  extension [L <: Label](t: Tensor1[L])

    def dot(other: Tensor1[L]): Tensor0 =
      Tensor0(Jax.jnp.dot(t.jaxValue, other.jaxValue))

    @targetName("tensor1MatmulTensor2")
    def matmul[L2 <: Label : ValueOf](other: Tensor2[L, L2]): Tensor1[L2] =
      Tensor(Jax.jnp.dot(t.jaxValue, other.jaxValue))

    def as[NewL <: Label](axis: Axis[NewL]): Tensor1[NewL] = t.relabel(Axis[L], axis)

  extension [L1 <: Label : ValueOf, L2 <: Label : ValueOf](t: Tensor2[L1, L2])

    def transpose: Tensor2[L2, L1] =
      Tensor(Jax.jnp.transpose(t.jaxValue))

    @targetName("tensor2MatmulTensor2")
    def matmul[L2Other <: Label : ValueOf](other: Tensor2[L2, L2Other]): Tensor2[L1, L2Other] =
      Tensor(Jax.jnp.matmul(t.jaxValue, other.jaxValue))

    @targetName("tensor2MatmulTensor1")
    def matmul1(other: Tensor1[L2]): Tensor1[L1] =
      Tensor(Jax.jnp.dot(t.jaxValue, other.jaxValue))
      
    @targetName("tensor2Det")
    def det: Tensor0 = Tensor0(Jax.jnp.linalg.det(t.jaxValue))

    @targetName("tensor2as")
    def as[NewL1 <: Label, NewL2 <: Label](newAxis1: Axis[NewL1], newAxis2: Axis[NewL2]): Tensor2[NewL1, NewL2] = 
      t.asInstanceOf[Tensor[(NewL1, NewL2)]]

  /** Tensor contraction operations
    *
    * Contract tensors over one or more shared axes using type-safe axis labels. This generalizes operations like dot
    * product, matrix multiplication, and tensor products.
    */
  extension [T <: Tuple](tensor: Tensor[T])

    /** Contract this tensor with another over a single shared axis
      *
      * Example: matrix-vector multiplication
      * {{{
      * val matrix = Tensor2[Rows, Features](...)
      * val vector = Tensor1[Features](...)
      * val result: Tensor1[Rows] = matrix.contract(Axis[Features])(vector)
      * }}}
      *
      * @param axis
      *   The axis to contract over (must exist in both tensors)
      * @param other
      *   The tensor to contract with
      * @return
      *   A new tensor with the contracted axis removed from both inputs
      */
    inline def contract[
        ContractAxis <: Label,
        OtherShape <: Tuple : NamesOf,
        ResultShape <: Tuple : NamesOf
    ](
        axis: Axis[ContractAxis]
    )(
        other: Tensor[OtherShape]
    )(
      using
      thisAxisIndex: AxisIndex[ContractAxis, T],
      otherAxisIndex: AxisIndex[ContractAxis, OtherShape],
    )(using
        // Ensure ContractAxis exists in both tensors
        ev1: Tuple.Contains[T, ContractAxis] =:= true,
        ev2: Tuple.Contains[OtherShape, ContractAxis] =:= true,
        // Compute result shape at compile time
        ev3: TupleHelpers.ContractResult[T, OtherShape, ContractAxis] =:= ResultShape
    ): Tensor[ResultShape] =
      import me.shadaj.scalapy.py.SeqConverters

      val axesTuple1 = Jax.Dynamic.global.tuple(Seq(thisAxisIndex.value).toPythonProxy)
      val axesTuple2 = Jax.Dynamic.global.tuple(Seq(otherAxisIndex.value).toPythonProxy)
      val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

      val result = Jax.jnp.tensordot(
        tensor.jaxValue,
        other.jaxValue,
        axes = axesPair
      )

      Tensor(result)
