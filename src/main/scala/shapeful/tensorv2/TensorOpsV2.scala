package shapeful.tensorv2
/*
import shapeful.Label
import shapeful.jaxv2.Jax
import shapeful.jaxv2.Einops
import scala.annotation.targetName
import scala.util.NotGiven
import Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.jax.Jax.PyDynamic
import TupleHelpers.{TreeOf, TreeOfImpl}
import scala.annotation.implicitNotFound
import TupleHelpers.{UnwrapAxes, TupleFlat}
import shapeful.tensorv2.TupleHelpers.DimExtractor

object TensorOpsV2:

  // -----------------------------------------------------------
  // 1. Elementwise Operations (The Field)
  // Preserves Shape: T -> T
  // -----------------------------------------------------------
  object Elementwise:
    extension [T <: Tuple : TreeOf](t: Tensor[T])
      
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

  // -----------------------------------------------------------
  // 2. Reduction Operations (The Monoid)
  // Reduces Rank: T -> T - {Axis}
  // -----------------------------------------------------------
  object Reduction:
    extension [T <: Tuple : TreeOf](t: Tensor[T])

      // Global Reductions (to Scalar)
      def sum: Tensor0 = Tensor0(Jax.jnp.sum(t.jaxValue))
      def mean: Tensor0 = Tensor0(Jax.jnp.mean(t.jaxValue))
      def max: Tensor0 = Tensor0(Jax.jnp.max(t.jaxValue))
      def min: Tensor0 = Tensor0(Jax.jnp.min(t.jaxValue))
      def argmax: Tensor0 = Tensor0(Jax.jnp.argmax(t.jaxValue))
      def argmin: Tensor0 = Tensor0(Jax.jnp.argmin(t.jaxValue))

      // Axis Reductions
      def sum[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import TreeOf.ForRemove.given
        Tensor(Jax.jnp.sum(t.jaxValue, axis = axisIndex.value))

      def mean[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import TreeOf.ForRemove.given
        Tensor(Jax.jnp.mean(t.jaxValue, axis = axisIndex.value))

      def max[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import TreeOf.ForRemove.given
        Tensor(Jax.jnp.max(t.jaxValue, axis = axisIndex.value))

      def min[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import TreeOf.ForRemove.given
        Tensor(Jax.jnp.min(t.jaxValue, axis = axisIndex.value))

      def argmax[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import TreeOf.ForRemove.given
        Tensor(Jax.jnp.argmax(t.jaxValue, axis = axisIndex.value))

      def argmin[ReduceAxis <: Label](
          axis: Axis[ReduceAxis]
      )(using axisIndex: AxisIndex[ReduceAxis, T]): Tensor[TupleHelpers.Remove[ReduceAxis, T]] =
        import TreeOf.ForRemove.given
        Tensor(Jax.jnp.argmin(t.jaxValue, axis = axisIndex.value))

  object LinearAlgebra:
    
    extension [T <: Tuple : TreeOf](tensor: Tensor[T])
      
      def outerProduct[OtherShape <: Tuple : TreeOf](other: Tensor[OtherShape]): Tensor[Tuple.Concat[T, OtherShape]] =
        import TreeOf.ForConcat.given
        import me.shadaj.scalapy.py.SeqConverters
        Tensor[Tuple.Concat[T, OtherShape]](
          // Jax outer product flattens, reshape required
          Jax.jnp.reshape(
            Jax.jnp.outer(tensor.jaxValue, other.jaxValue), 
            (tensor.shape.dimensions ++ other.shape.dimensions).toPythonProxy
          )
        )

      def contract[
          ContractAxis <: Label,
          OtherShape <: Tuple : TreeOf,
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
        import TreeOf.ForContractResult.given
        summon[TreeOf[TupleHelpers.ContractResult[T, OtherShape, ContractAxis]]]
        import me.shadaj.scalapy.py.SeqConverters

        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(thisAxisIndex.value).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(otherAxisIndex.value).toPythonProxy)
        val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

        val result = Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair)
        Tensor(result)

    extension [L <: Label](t: Tensor1[L])
      def dot(other: Tensor1[L]): Tensor0 =
        Tensor0(Jax.jnp.dot(t.jaxValue, other.jaxValue))

      @targetName("tensor1MatmulTensor2")
      def matmul[L2 <: Label : ValueOf](other: Tensor2[L, L2]): Tensor1[L2] =
        Tensor(Jax.jnp.dot(t.jaxValue, other.jaxValue))

    extension [L1 <: Label : ValueOf, L2 <: Label : ValueOf](t: Tensor2[L1, L2])
      @targetName("tensor2MatmulTensor2")
      def matmul[L2Other <: Label : ValueOf](other: Tensor2[L2, L2Other]): Tensor2[L1, L2Other] =
        Tensor(Jax.jnp.matmul(t.jaxValue, other.jaxValue))

      @targetName("tensor2MatmulTensor1")
      def matmul1(other: Tensor1[L2]): Tensor1[L1] =
        Tensor(Jax.jnp.dot(t.jaxValue, other.jaxValue))
      
      @targetName("tensor2Det")
      def det: Tensor0 = Tensor0(Jax.jnp.linalg.det(t.jaxValue))

  // -----------------------------------------------------------
  // 4. Structural Operations (Isomorphisms)
  // Permutations and Views: T1 -> T2 (Size(T1) == Size(T2))
  // -----------------------------------------------------------
  object Structural:
    
    // --- General ---
    extension (l: List[String])
      def removeAt(index: Int): List[String] = l.patch(index, Nil, 1)

    extension [T <: Tuple : TreeOf](tensor: Tensor[T])
      
      def rearrange[newT <: Tuple](
          newOrder: newT,
      )(
        using 
        evAllAxesInTensor: TupleFlat[UnwrapAxes[newT]] =:= TupleFlat[T],
        newNames: TreeOf[UnwrapAxes[newT]],
      ): Tensor[UnwrapAxes[newT]] = rearrange[newT, EmptyTuple](newOrder, EmptyTuple)

      def rearrange[newT <: Tuple, Dims <: Tuple](
          newOrder: newT,
          dims: Dims,
      )(
        using 
        evAllAxesInTensor: TupleFlat[UnwrapAxes[newT]] =:= TupleFlat[T],
        newNames: TreeOf[UnwrapAxes[newT]],
        extractor: DimExtractor[Dims],
      ): Tensor[UnwrapAxes[newT]] =
        
        def createEinopsPattern(fromPattern: String, toPattern: String): String =
          // Einops logic helper
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
          s"$fromCleaned -> $toCleaned"

        val fromPattern = tensor.shape.labelsTree.mkString(" ")
        val toPattern = newNames.tree.mkString(" ")
        val pattern = createEinopsPattern(fromPattern, toPattern)
        val dimSizesMap = extractor.extract(dims)
        
        Tensor(Einops.rearrange(tensor.jaxValue, pattern, kwargsMap = dimSizesMap))

    // --- Specific Structural Ops ---
    extension [L <: Label](t: Tensor1[L])
      def as[NewL <: Label](axis: Axis[NewL]): Tensor1[NewL] = t.relabel(Axis[L], axis)

    extension [L1 <: Label : ValueOf, L2 <: Label : ValueOf](t: Tensor2[L1, L2])
      def transpose: Tensor2[L2, L1] = Tensor(Jax.jnp.transpose(t.jaxValue))
      
      @targetName("tensor2as")
      def as[NewL1 <: Label, NewL2 <: Label](newAxis1: Axis[NewL1], newAxis2: Axis[NewL2]): Tensor2[NewL1, NewL2] = 
        t.asInstanceOf[Tensor[(NewL1, NewL2)]]

  // -----------------------------------------------------------
  // 5. Functional Operations (Higher Order)
  // Lifting functions over axes
  // -----------------------------------------------------------
  object Functional:
    extension [T <: Tuple : TreeOf](t: Tensor[T])
      def vmap[VmapAxis <: Label : ValueOf, OuterShape <: Tuple : TreeOf](
        axis: Axis[VmapAxis]
      )(
          f: Tensor[TupleHelpers.Remove[VmapAxis, T]] => Tensor[OuterShape]
      )(
        using 
        vmapAxisIndex: AxisIndex[VmapAxis, T],
      ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =
        import TreeOf.ForRemove.given
        val fpy = (jxpr: Jax.PyDynamic) =>
          val innerTensor = Tensor[TupleHelpers.Remove[VmapAxis, T]](jxpr)
          val result = f(innerTensor)
          result.jaxValue
        Tensor(Jax.jax_helper.vmap(fpy, vmapAxisIndex.value)(t.jaxValue))

  // -----------------------------------------------------------
  // 6. Scalar Operations (Primitives)
  // -----------------------------------------------------------
  object ScalarOps:
    extension (t: Tensor0)
      def toInt: Int = t.jaxValue.item().as[Int]
      def toFloat: Float = t.jaxValue.item().as[Float]
      def toBool: Boolean = t.jaxValue.item().as[Boolean]

      @targetName("tensor0Pow")
      def pow(exponent: Tensor0): Tensor0 = Tensor0(Jax.jnp.pow(t.jaxValue, exponent.jaxValue))

  // Export everything for the user
  export Elementwise.*
  export Reduction.*
  export LinearAlgebra.*
  export Structural.*
  export Functional.*
  export ScalarOps.*
  */