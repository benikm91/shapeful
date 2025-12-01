package shapeful.tensorv2

import shapeful.Label
import shapeful.jaxv2.Jax
import shapeful.jaxv2.Einops
import scala.annotation.targetName
import scala.util.NotGiven
import Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.jax.Jax.PyDynamic
import scala.annotation.implicitNotFound
import TupleHelpers.{UnwrapAxes, RemoveAll}
import shapeful.tensorv2.TupleHelpers.DimExtractor
import shapeful.tensorv2.TupleHelpers.ValuesOf.WrapAxes
import shapeful.tensorv2.TupleHelpers.NameOf

object TensorOps:

  // -----------------------------------------------------------
  // 1. Elementwise Operations (The Field)
  // Preserves Shape: T -> T
  // -----------------------------------------------------------
  object Elementwise:
    extension [T <: Tuple : NameOf](t: Tensor[T])
      
      // --- Basic Arithmetic ---
      @targetName("addTensor")
      def +(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.add(t.jaxValue, other.jaxValue))
      @targetName("addScalar")
      def +(other: Tensor0): Tensor[T] = Tensor(Jax.jnp.add(t.jaxValue, other.jaxValue))

      @targetName("subTensor")
      def -(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))
      @targetName("subScalar")
      def -(other: Tensor0): Tensor[T] = Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))

      @targetName("mulTensor")
      def *(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      @targetName("mulScalar")
      def *(other: Tensor0): Tensor[T] = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))

      @targetName("divTensor")
      def /(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))
      @targetName("divScalar")
      def /(other: Tensor0): Tensor[T] = Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))

      // --- Unary Math ---
      def abs: Tensor[T] = Tensor(Jax.jnp.abs(t.jaxValue))
      def sign: Tensor[T] = Tensor(Jax.jnp.sign(t.jaxValue))
      def pow(n: Tensor0): Tensor[T] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))
      def sqrt: Tensor[T] = Tensor(Jax.jnp.sqrt(t.jaxValue))
      def exp: Tensor[T] = Tensor(Jax.jnp.exp(t.jaxValue))
      def log: Tensor[T] = Tensor(Jax.jnp.log(t.jaxValue))
      def sin: Tensor[T] = Tensor(Jax.jnp.sin(t.jaxValue))
      def cos: Tensor[T] = Tensor(Jax.jnp.cos(t.jaxValue))
      def tanh: Tensor[T] = Tensor(Jax.jnp.tanh(t.jaxValue))

      // --- Clipping ---
      def clip(min: Float, max: Float): Tensor[T] = Tensor(Jax.jnp.clip(t.jaxValue, min, max))
      def clip(min: Tensor0, max: Tensor0): Tensor[T] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))

      // --- Comparison ---
      def <(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.less(t.jaxValue, other.jaxValue))
      def <=(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.less_equal(t.jaxValue, other.jaxValue))
      def >(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.greater(t.jaxValue, other.jaxValue))
      def >=(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.greater_equal(t.jaxValue, other.jaxValue))
  
  end Elementwise

  // -----------------------------------------------------------
  // 2. Reduction Operations (The Monoid)
  // Reduces Rank: T -> T - {Axis}
  // -----------------------------------------------------------
  object Reduction:

    trait SingleAxisReductionSupport[In, Axis]:
        type Out <: Tuple
        def index: Int
        def nameOf: NameOf[Out]

    object SingleAxisReductionSupport:
        given [In <: Tuple, Ax <: Label](using
            idx: AxisIndex[In, Ax],
            rm: Remover[In, Ax],
            nm: NameOf[rm.Out]
        ): SingleAxisReductionSupport[In, Ax] with
            type Out = rm.Out
            def index = idx.value
            def nameOf = nm

    extension [T <: Tuple : NameOf](t: Tensor[T])

        private def reduceOp[A <: Label](axis: Axis[A], opAny: Any)(using 
            support: SingleAxisReductionSupport[T, A]
        ): Tensor[support.Out] =
            val op = opAny.asInstanceOf[(PyDynamic, Int) => PyDynamic]
            Tensor(op(t.jaxValue, support.index))(using support.nameOf)

        // Global Reductions (to Scalar)
        def sum: Tensor0 = Tensor0(Jax.jnp.sum(t.jaxValue))
        def mean: Tensor0 = Tensor0(Jax.jnp.mean(t.jaxValue))
        def max: Tensor0 = Tensor0(Jax.jnp.max(t.jaxValue))
        def min: Tensor0 = Tensor0(Jax.jnp.min(t.jaxValue))
        def argmax: Tensor0 = Tensor0(Jax.jnp.argmax(t.jaxValue))
        def argmin: Tensor0 = Tensor0(Jax.jnp.argmin(t.jaxValue))

        def sum[A <: Label](axis: Axis[A])(using r: SingleAxisReductionSupport[T, A]) = t.reduceOp(axis, Jax.jnp.sum)
        def mean[A <: Label](axis: Axis[A])(using r: SingleAxisReductionSupport[T, A]) = t.reduceOp(axis, Jax.jnp.mean)
        def max[A <: Label](axis: Axis[A])(using r: SingleAxisReductionSupport[T, A]) = t.reduceOp(axis, Jax.jnp.max)
        def min[A <: Label](axis: Axis[A])(using r: SingleAxisReductionSupport[T, A]) = t.reduceOp(axis, Jax.jnp.min)
        def argmax[A <: Label](axis: Axis[A])(using r: SingleAxisReductionSupport[T, A]) = t.reduceOp(axis, Jax.jnp.argmax)
        def argmin[A <: Label](axis: Axis[A])(using r: SingleAxisReductionSupport[T, A]) = t.reduceOp(axis, Jax.jnp.argmin)
    
  end Reduction

  object Contraction:

    extension [T <: Tuple : NameOf](tensor: Tensor[T])
      
      def outerProduct[OtherShape <: Tuple : NameOf](other: Tensor[OtherShape]): Tensor[Tuple.Concat[T, OtherShape]] =
        import me.shadaj.scalapy.py.SeqConverters
        import NameOf.ForConcat.given
        Tensor(
          // Jax outer product flattens, reshape required
          Jax.jnp.reshape(
            Jax.jnp.outer(tensor.jaxValue, other.jaxValue), 
            (tensor.shape.dimensions ++ other.shape.dimensions).toPythonProxy
          )
        )

      def contract[
          ContractAxis <: Label : ValueOf,
          OtherShape <: Tuple : NameOf,
      ]
      (axis: Axis[ContractAxis])
      (other: Tensor[OtherShape])(using
        axisIndex: AxisIndex[T, ContractAxis],
        remover: Remover[T, ContractAxis],
        otherAxisIndex: AxisIndex[OtherShape, ContractAxis],
        otherRemover: Remover[OtherShape, ContractAxis],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import me.shadaj.scalapy.py.SeqConverters
        import NameOf.ForConcat.given

        given removerNameOf: NameOf[remover.Out] = NameOf.removerNameOf(remover)
        given otherRemoverNameOf: NameOf[otherRemover.Out] = NameOf.removerNameOf(otherRemover)

        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(axisIndex.value).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(otherAxisIndex.value).toPythonProxy)
        val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)
        Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))
  
  end Contraction

  object LinearAlgebra:
    extension [T <: Tuple : NameOf](t: Tensor[T])
      def det: Tensor0 = Tensor0(Jax.jnp.linalg.det(t.jaxValue))
      def norm: Tensor0 = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))
      def inv: Tensor[T] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))
      def trace: Tensor0 = Tensor0(Jax.jnp.trace(t.jaxValue))

  end LinearAlgebra

  // -----------------------------------------------------------
  // 4. Structural Operations (Isomorphisms)
  // Permutations and Views: T1 -> T2 (Size(T1) == Size(T2))
  // -----------------------------------------------------------
  object Structural:
    
    type ExtractLabel[X] = X match
        case (Axis[l], Int) => l

    type ExtractLabels[Inputs <: Tuple] = Tuple.Map[Inputs, ExtractLabel]

    extension [T <: Tuple : NameOf](tensor: Tensor[T])
      
      def slice[L <: Label](
        axisWithSliceIndex: (Axis[L], Int),
      )(using 
        remover: Remover[T, L],
        axisIndex: AxisIndex[T, L],
        namesOf: NameOf[L *: EmptyTuple],
      ): Tensor[remover.Out] = slice(Tuple1(axisWithSliceIndex))


      def slice[Inputs <: Tuple](
        axesWithSliceIndices: Inputs,
      )(using 
        remover: RemoverAll[T, ExtractLabels[Inputs]],
        axesIndices: AxisIndices[T, ExtractLabels[Inputs]],
        namesOf: NameOf[ExtractLabels[Inputs]],
      ): Tensor[remover.Out] =
        import me.shadaj.scalapy.py
        import me.shadaj.scalapy.py.SeqConverters

        val PySlice = py.Dynamic.global.slice
        val Colon = PySlice(py.None)

        val rank = tensor.shape.rank
        val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

        val targetDims: List[Int] = axesIndices.values
        val inputs = axesWithSliceIndices.toList.asInstanceOf[List[(Any, Int)]]

        targetDims.zip(inputs).foreach { 
            case (dimIndex, (axisObj, sliceIndex)) =>
                val dimSize = tensor.shape.dimensions(dimIndex)
                require(sliceIndex >= 0 && sliceIndex < dimSize, s"Slice index $sliceIndex out of bounds for dimension $dimIndex (size $dimSize)")
                indicesBuffer(dimIndex) = py.Any.from(sliceIndex)
        }

        val indexTuple = Jax.Dynamic.global.tuple(indicesBuffer.toSeq.toPythonProxy)
        val result = tensor.jaxValue.bracketAccess(indexTuple)

        given outputNameOf: NameOf[remover.Out] = NameOf.removerAllNameOf(remover)
        Tensor[remover.Out](result)
        

      def rearrange[newT <: Tuple](
          newOrder: newT,
      )(
        using 
        // evAllAxesInTensor: TupleFlat[UnwrapAxes[newT]] =:= TupleFlat[T],
        newNames: NameOf[UnwrapAxes[newT]],
      ): Tensor[UnwrapAxes[newT]] = rearrange[newT, EmptyTuple](newOrder, EmptyTuple)

      def rearrange[newT <: Tuple, Dims <: Tuple](
          newOrder: newT,
          dims: Dims,
      )(
        using 
        // evAllAxesInTensor: TupleFlat[UnwrapAxes[newT]] =:= TupleFlat[T],
        newNames: NameOf[UnwrapAxes[newT]],
        extractor: DimExtractor[Dims],
      ): Tensor[UnwrapAxes[newT]] =
        def createEinopsPattern(fromPattern: String, toPattern: String): String =
          def cleanPattern(pattern: String): String =
            // to replace all a*b*c in pattern with (a b c), example:
            // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
            val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
            regex.replaceAllIn(pattern, m => {
              val group = m.group(1)
              val replaced = group.split("\\*").mkString("(", " ", ")")
              replaced
            })
          s"${cleanPattern(fromPattern)} -> ${cleanPattern(toPattern)}"
        val fromPattern = tensor.shape.labels.mkString(" ")
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

      def as[newT <: Tuple](
        using 
        newNames: NameOf[UnwrapAxes[newT]],
        @implicitNotFound("Cannot convert tensor of shape ${T} to shape ${newT} due to size mismatch.")
        evSameSize: Tuple.Size[newT] =:= Tuple.Size[T],
      ): Tensor[UnwrapAxes[newT]] = Tensor[UnwrapAxes[newT]](tensor.jaxValue)
  
  end Structural

  // -----------------------------------------------------------
  // 5. Functional Operations (Higher Order)
  // Lifting functions over axes
  // -----------------------------------------------------------
  object Functional:
    extension [T <: Tuple : NameOf](t: Tensor[T])
      
      def vmap[VmapAxis <: Label : ValueOf, OuterShape <: Tuple : NameOf](
        axis: Axis[VmapAxis]
      )(using
        remover: Remover[T, VmapAxis],
        axisIndex: AxisIndex[T, VmapAxis],
        vmapAxisIndex: AxisIndex[T, VmapAxis],
      )(
          f: Tensor[remover.Out] => Tensor[OuterShape]
      ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =
        val fpy = (jxpr: Jax.PyDynamic) =>
            given namesOf: NameOf[remover.Out] = NameOf.removerNameOf(remover)
            val innerTensor = Tensor[remover.Out](jxpr)
            val result = f(innerTensor)
            result.jaxValue

        Tensor(Jax.jax_helper.vmap(fpy, vmapAxisIndex.value)(t.jaxValue))

  end Functional

  export Elementwise.*
  export Reduction.*
  export Contraction.*
  export LinearAlgebra.*
  export Structural.*
  export Functional.*

  // -----------------------------------------------------------
  // Common specialized operation names
  // -----------------------------------------------------------
  object ScalarOps:
    extension (t: Tensor0)
      def toInt: Int = t.jaxValue.item().as[Int]
      def toFloat: Float = t.jaxValue.item().as[Float]
      def toBool: Boolean = t.jaxValue.item().as[Boolean]

      @targetName("tensor0Pow")
      def pow(exponent: Tensor0): Tensor0 = Tensor0(Jax.jnp.pow(t.jaxValue, exponent.jaxValue))

  object VectorOps:

    extension [L <: Label : ValueOf](t: Tensor1[L])
      def dot(other: Tensor1[L]): Tensor0 = t.innerDot(other)
      def innerDot(other: Tensor1[L]): Tensor0 = t.contract(Axis[L])(other)
      def outerDot[OtherLabel <: Label : ValueOf](other: Tensor1[OtherLabel]): Tensor2[L, OtherLabel] = 
        t.outerProduct(other)

  object MatrixOps:
    extension [L1 <: Label : ValueOf, L2 <: Label : ValueOf](t: Tensor2[L1, L2])
      def transpose: Tensor2[L2, L1] = t.rearrange((Axis[L2], Axis[L1]))

      @targetName("tensor2MatmulTensor2")
      def matmul[L3 <: Label : ValueOf](other: Tensor2[L2, L3])(
        using 
        remover: Remover[(L1, L2), L2],
        otherRemover: Remover[(L2, L3), L2],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import NameOf.ForConcat.given
        t.contract(Axis[L2])(other)

      @targetName("tensor2MatmulTensor1")
      def matmul(other: Tensor1[L2])(
        using 
        remover: Remover[(L1, L2), L2],
        otherRemover: Remover[Tuple1[L2], L2],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import NameOf.ForConcat.given
        t.contract(Axis[L2])(other)
      
  
  export ScalarOps.*
  export VectorOps.*
  export MatrixOps.*

end TensorOps

object StatisticOps:

  extension [T <: Tuple : NameOf](t: Tensor[T])
    def std: Tensor0 = Tensor0(Jax.jnp.std(t.jaxValue))

end StatisticOps