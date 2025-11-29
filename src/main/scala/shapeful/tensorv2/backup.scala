package shapeful.tensorv2

/*

package shapeful.tensorv2

import shapeful.Label
import shapeful.jaxv2.Jax
import shapeful.jaxv2.Einops
import scala.annotation.targetName
import scala.util.NotGiven
import Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.jax.Jax.PyDynamic
import TupleHelpers.{ShapeTreeOf, ShapeTreeOfImpl}
import scala.annotation.implicitNotFound
import TupleHelpers.{UnwrapAxes, TupleFlat}
import shapeful.tensorv2.TupleHelpers.DimExtractor

object TensorOps:

  extension (l: List[String])
    def removeAt(index: Int): List[String] = l.patch(index, Nil, 1)

  object Mapping:

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

  object BasicUniaryOperations:

    extension [T <: Tuple : ShapeTreeOf](t: Tensor[T])

      def norm: Tensor0 = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))

      def sum: Tensor0 = Tensor0(Jax.jnp.sum(t.jaxValue))
      def sum[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import ShapeTreeOf.ForRemove.given
        Tensor(Jax.jnp.sum(t.jaxValue, axis = axisIndex.value))

      def mean: Tensor0 = Tensor0(Jax.jnp.mean(t.jaxValue))
      def mean[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import ShapeTreeOf.ForRemove.given
        Tensor(Jax.jnp.mean(t.jaxValue, axis = axisIndex.value))

      def max: Tensor0 = Tensor0(Jax.jnp.max(t.jaxValue))
      def max[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import ShapeTreeOf.ForRemove.given
        Tensor(Jax.jnp.max(t.jaxValue, axis = axisIndex.value))

      def min: Tensor0 = Tensor0(Jax.jnp.min(t.jaxValue))
      def min[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import ShapeTreeOf.ForRemove.given
        Tensor(Jax.jnp.min(t.jaxValue, axis = axisIndex.value))

      def argmax: Tensor0 = Tensor0(Jax.jnp.argmax(t.jaxValue))
      def argmax[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import ShapeTreeOf.ForRemove.given
        Tensor(Jax.jnp.argmax(t.jaxValue, axis = axisIndex.value))

      def argmin: Tensor0 = Tensor0(Jax.jnp.argmin(t.jaxValue))
      def argmin[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import ShapeTreeOf.ForRemove.given
        Tensor(Jax.jnp.argmin(t.jaxValue, axis = axisIndex.value))

      def exp: Tensor[T] = Tensor(Jax.jnp.exp(t.jaxValue))
      def log: Tensor[T] = Tensor(Jax.jnp.log(t.jaxValue))

      def pow(n: Tensor0): Tensor[T] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))

  object BasicBinaryOperations:

    extension [T <: Tuple : ShapeTreeOf](t: Tensor[T])

      def +(other: Tensor[T]): Tensor[T] =
        Tensor(Jax.jnp.add(t.jaxValue, other.jaxValue))

      @targetName("tensor0PlusScalar")
      def +(other: Tensor0): Tensor[T] =
        Tensor(Jax.jnp.add(t.jaxValue, other.jaxValue))

      // Subtraction with same shape - most common case
      def -(other: Tensor[T]): Tensor[T] =
        Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))

      @targetName("tensor0MinusScalar")
      def -(other: Tensor0): Tensor[T] =
        Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))

      def *(other: Tensor[T]): Tensor[T] =
        Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))

      @targetName("tensor0MultScalar")
      def *(other: Tensor0): Tensor[T] =
        Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      
      @targetName("tensorDivTensor")
      def /(other: Tensor[T]): Tensor[T] =
        Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))

      @targetName("tensorDivScalar")
      def /(other: Tensor0): Tensor[T] =
        Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))

    def std: Tensor0 = Tensor0(Jax.jnp.std(t.jaxValue))

    def variance: Tensor0 = Tensor0(Jax.jnp.`var`(t.jaxValue))

    def std[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import ShapeTreeOf.ForRemove.given
      Tensor(Jax.jnp.std(t.jaxValue, axis = axisIndex.value))

    def variance[ReduceAxis <: Label](
        axis: Axis[ReduceAxis]
    )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
      import ShapeTreeOf.ForRemove.given
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

  export Mapping.*
  export BasicUniaryOperations.*

  object Tensor0Ops:

    extension (t: Tensor0)
      def toInt: Int = t.jaxValue.item().as[Int]
      def toFloat: Float = t.jaxValue.item().as[Float]
      def toBool: Boolean = t.jaxValue.item().as[Boolean]

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

  extension [T <: Tuple : ShapeTreeOf](tensor: Tensor[T])

    def contract[
        ContractAxis <: Label,
        OtherShape <: Tuple : ShapeTreeOf,
    ](
        axis: Axis[ContractAxis]
    )(
      using 
      @implicitNotFound("Axis ${ContractAxis} not found in tensor of shape ${T}")
      evAxisInTensor: Tuple.Contains[T, ContractAxis] =:= true,
    )(
        other: Tensor[OtherShape]
    )(
        using
        @implicitNotFound("Axis ${ContractAxis} not found in tensor of shape ${OtherShape}")
        evAxisInOther: Tuple.Contains[OtherShape, ContractAxis] =:= true,
    )(
      using
      thisAxisIndex: AxisIndex[ContractAxis, T],
      otherAxisIndex: AxisIndex[ContractAxis, OtherShape],
    ): Tensor[TupleHelpers.ContractResult[T, OtherShape, ContractAxis]] =
      import ShapeTreeOf.ForContractResult.given
      summon[ShapeTreeOf[TupleHelpers.ContractResult[T, OtherShape, ContractAxis]]]
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

    def outerProduct[OtherShape <: Tuple : ShapeTreeOf](other: Tensor[OtherShape]): Tensor[Tuple.Concat[T, OtherShape]] =
      import ShapeTreeOf.ForConcat.given
      import me.shadaj.scalapy.py.SeqConverters
      Tensor[Tuple.Concat[T, OtherShape]](
        // Jax outer product flattens the result, so we need to reshape it back to the original shape
        Jax.jnp.reshape(
          Jax.jnp.outer(
            tensor.jaxValue,
            other.jaxValue,
          ), 
          (tensor.shape.dimensions ++ other.shape.dimensions).toPythonProxy
        )
      )

    def rearrange[newT <: Tuple](
        newOrder: newT,
    )(
      using 
      evAllAxesInTensor: TupleFlat[UnwrapAxes[newT]] =:= TupleFlat[T],
      newNames: ShapeTreeOf[UnwrapAxes[newT]],
    ): Tensor[UnwrapAxes[newT]] = rearrange[newT, EmptyTuple](newOrder, EmptyTuple)

    def rearrange[newT <: Tuple, Dims <: Tuple](
        newOrder: newT,
        dims: Dims,
    )(
      using 
      evAllAxesInTensor: TupleFlat[UnwrapAxes[newT]] =:= TupleFlat[T],
      newNames: ShapeTreeOf[UnwrapAxes[newT]],
      extractor: DimExtractor[Dims],
    ): Tensor[UnwrapAxes[newT]] =
      def createEinopsPattern(fromPattern: String, toPattern: String): String =
        /** 
         * Replace shared groups like "... (w h) ... -> ..., (w h) ..." with dummy "... w_h ... -> ... w_h ...""
         * Necessary because einops does not support shared groups in from and to patterns.
         */ 
        def replaceSharedGroups(fromPattern: String, toPattern: String): (String, String) =
          def tokenize(s: String): List[String] = 
            val tokenRegex = """\([^)]+\)|\S+""".r
            tokenRegex.findAllIn(s).toList
          def findGroups(tokens: List[String]): Set[String] = 
            tokens.filter(str => str.startsWith("(") && str.endsWith(")")).toSet
          val fromTokens = tokenize(fromPattern)
          val toTokens = tokenize(toPattern)
          val fromGroups = findGroups(fromTokens)
          val toGroups = findGroups(toTokens)
          val sharedGroups = fromGroups.intersect(toGroups)
          val aliases = sharedGroups.map { groupStr =>
            val cleanName = groupStr.replaceAll("[()]", "").trim.replaceAll("\\s+", "_")
            groupStr -> cleanName
          }.toMap
          val fromCleaned = fromTokens.map(t => aliases.getOrElse(t, t)).mkString(" ")
          val toCleaned   = toTokens.map(t => aliases.getOrElse(t, t)).mkString(" ")
          (fromCleaned, toCleaned)
        val (fromCleaned, toCleaned) = replaceSharedGroups(fromPattern, toPattern)
        println(s"Einops rearrange pattern: $fromCleaned -> $toCleaned")
        s"$fromCleaned -> $toCleaned"
      val fromPattern = tensor.shape.labelsTree.mkString(" ")
      val toPattern = newNames.tree.mkString(" ")
      val pattern = createEinopsPattern(fromPattern, toPattern)
      val dimSizesMap = extractor.extract(dims)
      Tensor(
        Einops.rearrange(
          tensor.jaxValue,
          pattern,
          kwargsMap = dimSizesMap
        )
      )

*/
