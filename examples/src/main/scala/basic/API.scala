package src.main.scala.basic

import shapeful.StringMath.*
import shapeful.tensorv2.{Axis, Shape, Tensor1, Tensor2, Tensor, DType, Device}
import scala.collection.compat.immutable.ArraySeq
import shapeful.tensorv2.TensorOps.*
import shapeful.tensorv2.TupleHelpers
import shapeful.tensorv2.TupleHelpers.NameOf
import shapeful.tensorv2.TupleHelpers.UnwrapAxes
import shapeful.tensorv2.TupleHelpers.ValuesOf
import shapeful.tensorv2.AxisIndex
import shapeful.tensorv2.TupleHelpers.ValuesOf.AxesFactory
import shapeful.tensorv2.Remover
import scala.collection.View.Zip

def main(args: Array[String]): Unit =
  val AB = Tensor.ones(Shape(
    Axis["A"] -> 10,
    Axis["B"] -> 5,
  ))
  val AC = Tensor.ones(Shape(
      Axis["A"]-> 10,
      Axis["C"] -> 5
  ))
  val ABCD = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
      Axis["D"] -> 5,
  ))
  {
    import shapeful.tensorv2.Tensor0
    /** 
     * ELEMENT-WISE OPERATIONS
     */
    AB + AB
    AB * AB
    AB - AB
    AB / AB
    AB.abs
    AB.sign
    AB.pow(Tensor0(2))
    AB.sqrt
    AB.exp
    AB.log
    AB.sin
    AB.cos
    AB.tanh
    AB.clip(0, 1)
    AB < AB
    AB > AB
    AB <= AB
    AB >= AB
    AB == AB
    /** 
     * REDUCTION
     */
    val sum = AB.sum
    val sumAB_A = AB.sum(Axis["A"])
    val sumABCD_A = ABCD.sum(Axis["A"])
    val sumABCD_AB = ABCD.sum((Axis["A"], Axis["B"]))
    AB.mean
    AB.mean(Axis["A"])
    AB.max
    AB.max(Axis["A"])
    AB.min
    AB.min(Axis["A"])
    AB.argmax
    AB.argmax(Axis["A"])
    AB.argmin
    /** 
     * CONTRACT
     * Analog to JAX tensordot with a single axis, with two changes:
     * - Only a single axis is allowed TODO allow multiple axes
     */
    val resContract = AB.contract(Axis["A"])(AC)
    // note there are some matrix specific contraction
    val resMatrixMultiply = AB.transpose.matmul(AC)
    /** 
     * OUTER PRODUCT (contract over zero axes)
     * Analog to JAX outer product, i.e., no axes to contract
     */
    val resOuterProduct = AB.outerProduct(AC)
    /** 
     * SLICE 
     * Analog to JAX slice(...) or JAX at(...).get, with two changes:
     * - Out of range index leads to an error (instead of clipping)
     * - No colon access (e.g., X[:, 0]), as due to name of axes this is not necessary (just leave out name)
     */
    // Select single index
    val resSliceA0 = AB.slice(Axis["A"] -> 0)
    // Select range AB[0:1, :]
    val resSliceA01 = AB.slice(Axis["A"] -> (0 until 1))
    // Select range and index AB[0:1, 2]
    val resSliceA01B2 = AB.slice((  // TODO make (()) optional
        Axis["A"] -> (0 until 1),
        Axis["B"] -> 2,
    ))
    // Select list of indices AB[[0,3,6], :]
    val resSliceList = AB.slice((
      Axis["A"] -> List(0, 3, 6),
    ))
    /** 
     * SET 
     * Analog to JAX at(...).set, with two changes:
     * - Out of range index leads to an error (instead of clipping)
     * - No colon access (e.g., X[:, 0]), as due to name of axes this is not necessary (just leave out name)
     */
    // set row vector at A=0, AB.at[0, :].set([0,1,2,3,4])
    val resSetA0 = AB.set(
      Axis["A"] -> 0
    )(Tensor1(Axis["B"], ArraySeq(0, 1, 2, 3, 4))) // Tensor2[("A", "B")]
    // set sub-matrix, AB.at[0:1, 0:1].set([[1,2],[3,4]])
    val resSetA01B01 = AB.set((  // TODO make (()) optional
      Axis["A"] -> (0 to 1),
      Axis["B"] -> (0 to 1),
    ))(Tensor2(
        Axis["A"], 
        Axis["B"],
        ArraySeq(
          ArraySeq(1f, 2f), 
          ArraySeq(3f, 4f),
        )
    ))
    /**
     * REARRANGE
     * Analog to einops rearrange, but with named axes. For JAX this replaces `transpose` and `reshape` operations.
     */
    // einops.rearrange(ABCD, 'a b c d -> b a c d')
    val resRearrangeABCDSwap = ABCD.rearrange((
        Axis["B"],
        Axis["A"],
        Axis["C"],
        Axis["D"],
    ))
    // einops.rearrange(ABCD, 'a b c d -> (b a) c d')
    val resRearrangeABCDFlat = ABCD.rearrange((  // TODO make Tuple1 optional
      Axis["B" * "A"],
      Axis["C"],
      Axis["D"],
    ))
    /** AS - rename axes labels */
    // no JAX equivalent as axes are not named in JAX
    val resAsBA = AB.as[(Axis["X"], Axis["Y"])]
    /** SWAP */
    // AB.swapaxes(1, 0)
    val resSwap = AB.swap(Axis["A"], Axis["B"])
    /** RAVEL */
    // AB.ravel()
    val resRavel = ABCD.ravel // Tensor1[("A*B")]
    /** 
     * APPEND AXIS
     * Analog to jnp.expand_dims / None indexing in JAX, adds a new axis at the end or beginning.
     * If axis must be inserted at a specific position use `rearrange` after `appendAxis` or `prependAxis`.
     */
    // AB[:, :, None]
    val resAppendAxis = AB.appendAxis(Axis["C"]) // Tensor3[("A", "B", "C")]
    /** 
     * PREPEND AXIS
     * Analog to jnp.expand_dims / None indexing in JAX
     */
    // AB[None, :, :]
    val resPrependAxis = AB.prependAxis(Axis["C"]) // Tensor3[("C", "A", "B")]
    /** 
     * SQUEEZE
     * Analog to jnp.squeeze in JAX
     */
    // AB.squeeze(axis=0)
    val resSqueeze = Tensor.ones(Shape(
      Axis["A"] -> 1,
      Axis["B"] -> 3,
    )).squeeze(Axis["A"]) // Tensor1[("B")]
    /** VMAP (/ ZIPVMAP)
     * Analog to JAX vmap, with one changes:
     * - vmap allows only single axis
     * - zipvmap for multiple tensors to be mapped over the same axis (vmap in JAX)
     */
    val resVmapAB = AB.vmap(Axis["A"]){ row => row.sum }
    val resVmapABCD = ABCD.vmap(Axis["C"]){ sliceABD => 
      val resBD = sliceABD.sum(Axis["A"]) 
      resBD
    }
    val resZipVmap2 = zipvmap(Axis["A"])((AB, AC)) { 
      case (abi, aci) => abi.sum + aci.sum 
    }
    val resZipVmap4 = zipvmap(Axis["A"])((AB, AC, AB, AC)) { 
      case (abi, aci, ab2i, ac2i) => abi.sum + aci.sum + ab2i.sum + ac2i.sum
    }
  }
  {
    /**
     * WHERE 
     * Analog to jnp.where in JAX
     */
    val x = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val y = Tensor.zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val condition = Tensor.zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).asType(DType.Bool)
    val res = where(condition, x, y)
  }
  {
    /**
      * LINEAR ALGEBRA
      */
    val X = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 2,
    ))
    X.det   // Analog to jnp.linalg.det
    X.norm  // Analog to jnp.linalg.norm
    X.inv   // Analog to jnp.linalg.inv
    X.trace // Analog to jnp.trace
  }
  {
    /**
     * MATRIX  
     */
    AB.diagonal // Analog to jnp.diagonal
  }