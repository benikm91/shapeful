package shapeful.tensorv2

import shapeful.Label
import shapeful.jaxv2.Jax
import scala.annotation.targetName
import scala.util.NotGiven
import Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.jax.Jax.PyDynamic
import TupleHelpers.{ShapeTreeOf, ShapeTreeOfImpl}

object TensorOpsV2:

  extension (l: List[String])
    def removeAt(index: Int): List[String] = l.patch(index, Nil, 1)

  extension [T <: Tuple : ShapeTreeOf](t: Tensor[T])

    def vmap[VmapAxis <: Label : ValueOf, OuterShape <: Tuple : ShapeTreeOf](
      axis: Axis[VmapAxis]
    )(
        f: Tensor[TupleHelpers.Remove[VmapAxis, T]] => Tensor[OuterShape]
    )(
      using 
      vmapAxisIndex: AxisIndex[VmapAxis, T],
    ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =
      import ShapeTreeOf.ForRemove.given
      val fpy = (jxpr: Jax.PyDynamic) =>
        val innerTensor = Tensor[TupleHelpers.Remove[VmapAxis, T]](jxpr)
        val result = f(innerTensor)
        result.jaxValue
      Tensor(Jax.jax_helper.vmap(fpy, vmapAxisIndex.value)(t.jaxValue))

  extension [T <: Tuple](tensor: Tensor[T])

    def contract[
        ContractAxis <: Label,
        OtherShape <: Tuple : ShapeTreeOf,
        ResultShape <: Tuple : ShapeTreeOf
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
