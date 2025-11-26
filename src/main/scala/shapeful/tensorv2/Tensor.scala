package shapeful.tensorv2

import scala.annotation.targetName
import scala.collection.immutable.ArraySeq
import scala.compiletime.{erasedValue, summonFrom}
import shapeful.jaxv2.Jax
import shapeful.jaxv2.JaxDType
import shapeful.jaxv2.Jax.PyDynamic
import shapeful.Label
import shapeful.tensor.TupleHelpers.{IntTuple, StringTuple}
import shapeful.random.Random
import me.shadaj.scalapy.py.SeqConverters

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  // case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])

object Device:
  val default: Device = Device.CPU

case class Tensor[T <: Tuple] private[tensorv2] (
  val jaxValue: Jax.PyDynamic,
  val axes: List[String],
):

  // inline def names: List[String] = TupleHelpers.namesOf[T]

  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)

  lazy val shape: Shape[T] =
    new Shape[T](
      jaxValue.shape.as[Seq[Int]].toList,
      axes
    )

  def asType(newDType: DType): Tensor[T] = 
    copy(
      jaxValue = Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(newDType)), 
    )

  def toDevice(newDevice: Device): Tensor[T] = 
    copy(
      jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice), 
    )

  def reshape[NewT <: Tuple](newShape: Shape[NewT]): Tensor[NewT] =
    require(shape.size == newShape.size, "New shape must have the same number of elements")
    copy(
      jaxValue=Jax.jnp.reshape(jaxValue, newShape.dimensions.toPythonProxy),
      axes=newShape.labels,
    )

  def relabel[From <: Label, To <: Label](from: Axis[From], to: Axis[To]): Tensor[TupleHelpers.Replace[T, From, To]] =
    copy(
      axes = axes.map(
        label =>
          if label == from.name then
            to.name
          else
            label
      )
    )

  inline def rearrange[A1 <: Label](a1: Axis[A1])(using
      ev: Tuple.Size[Tuple1[A1]] =:= Tuple.Size[T]
  ): Tensor[Tuple1[A1]] =
    rearrangeImpl[Tuple1[A1]](Seq(getLabelForRearrange[A1](0)))

  inline def rearrange[A1 <: Label, A2 <: Label](
      a1: Axis[A1],
      a2: Axis[A2]
  )(using ev: Tuple.Size[(A1, A2)] =:= Tuple.Size[T]): Tensor[(A1, A2)] =
    rearrangeImpl[(A1, A2)](Seq(getLabelForRearrange[A1](0), getLabelForRearrange[A2](1)))

  inline def rearrange[A1 <: Label, A2 <: Label, A3 <: Label](
      a1: Axis[A1],
      a2: Axis[A2],
      a3: Axis[A3]
  )(using ev: Tuple.Size[(A1, A2, A3)] =:= Tuple.Size[T]): Tensor[(A1, A2, A3)] =
    rearrangeImpl[(A1, A2, A3)](
      Seq(getLabelForRearrange[A1](0), getLabelForRearrange[A2](1), getLabelForRearrange[A3](2))
    )

  inline def rearrange[A1 <: Label, A2 <: Label, A3 <: Label, A4 <: Label](
      a1: Axis[A1],
      a2: Axis[A2],
      a3: Axis[A3],
      a4: Axis[A4]
  )(using ev: Tuple.Size[(A1, A2, A3, A4)] =:= Tuple.Size[T]): Tensor[(A1, A2, A3, A4)] =
    rearrangeImpl[(A1, A2, A3, A4)](
      Seq(
        getLabelForRearrange[A1](0),
        getLabelForRearrange[A2](1),
        getLabelForRearrange[A3](2),
        getLabelForRearrange[A4](3)
      )
    )

  inline def rearrange[A1 <: Label, A2 <: Label, A3 <: Label, A4 <: Label, A5 <: Label](
      a1: Axis[A1],
      a2: Axis[A2],
      a3: Axis[A3],
      a4: Axis[A4],
      a5: Axis[A5]
  )(using ev: Tuple.Size[(A1, A2, A3, A4, A5)] =:= Tuple.Size[T]): Tensor[(A1, A2, A3, A4, A5)] =
    rearrangeImpl[(A1, A2, A3, A4, A5)](
      Seq(
        getLabelForRearrange[A1](0),
        getLabelForRearrange[A2](1),
        getLabelForRearrange[A3](2),
        getLabelForRearrange[A4](3),
        getLabelForRearrange[A5](4)
      )
    )

  private inline def getLabelForRearrange[L <: Label](idx: Int): String =
    scala.compiletime.summonFrom {
      case v: ValueOf[L] => v.value.toString
      case _             =>
        assert (idx >= 0 && idx < shape.labels.length, s"Index $idx out of bounds for shape labels")
        shape.labels(idx)
    }

  private def rearrangeImpl[NewOrder <: Tuple](newLabelNames: Seq[String]): Tensor[NewOrder] =
    val permutation = newLabelNames.map { labelName =>
      val idx = shape.labels.indexOf(labelName)
      require(idx >= 0, s"Axis ${labelName} not found in tensor shape")
      idx
    }.toList

    val transposedJax = Jax.jnp.transpose(jaxValue, permutation.toArray.toPythonProxy)
    val transposedAxes = permutation.map(i => shape.labels(i))

    copy(
      jaxValue = transposedJax,
      axes = transposedAxes,
    )

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [ _ ] =>> Int]

  def at(idx: IndicesOf[T]): TensorIndexer[T] =
    new TensorIndexer(this, idx)

  def tensorEquals(other: Tensor[?]): Boolean =
    Jax.jnp.array_equal(this.jaxValue, other.jaxValue).item().as[Boolean]

  def ==(other: Tensor[?]): Boolean = tensorEquals(other)

  def !=(other: Tensor[?]): Boolean = !(this == other)

  def elementEquals[U <: Tuple](other: Tensor[U])(
    using ev: Tuple.Size[T] =:= Tuple.Size[U]
  ): Tensor[T] =
    require(this.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${this.shape.dimensions} vs ${other.shape.dimensions}")
    copy(
      jaxValue = Jax.jnp.equal(this.jaxValue, other.jaxValue),
    )

  def approxEquals[U <: Tuple](other: Tensor[U], tolerance: Float = 1e-6f)(
    using ev: Tuple.Size[T] =:= Tuple.Size[U]
  ): Boolean =
    val result = Jax.jnp.allclose(
      this.jaxValue,
      other.jaxValue,
      atol = tolerance,
      rtol = tolerance
    )
    result.item().as[Boolean]

  override def equals(obj: Any): Boolean = obj match
    case other: Tensor[?] => this.tensorEquals(other)
    case _                => false

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

object Tensor:

  type Tensor0 = Tensor[EmptyTuple]
  type Tensor1[L <: Label] = Tensor[Tuple1[L]]
  type Tensor2[L1 <: Label, L2 <: Label] = Tensor[(L1, L2)]
  type Tensor3[L1 <: Label, L2 <: Label, L3 <: Label] = Tensor[(L1, L2, L3)]
  type Tensor4[L1 <: Label, L2 <: Label, L3 <: Label, L4 <: Label] = Tensor[(L1, L2, L3, L4)]

  def apply[T <: Tuple](shape: Shape[T], values: ArraySeq[Float], dtype: DType = DType.Float32, device: Device = Device.default): Tensor[T] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val jaxValues = Jax.jnp
      .array(
        values.toPythonProxy,
        dtype = JaxDType.jaxDtype(dtype),
        device = device.jaxDevice,
      )
      .reshape(shape.dimensions.toPythonProxy)
    new Tensor[T](jaxValues, shape.labels)

  def zeros[T <: Tuple](shape: Shape[T], dtype: DType): Tensor[T] =
    val jaxValues = Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
    new Tensor[T](jaxValues, shape.labels).asType(dtype)

  def zeros[L1 <: Label](axis1: (Axis[L1], Int)): Tensor1[L1] =
    zeros(Shape(axis1), DType.Float32)

  inline def zeros[L1 <: Label, L2 <: Label](
      axis1: (Axis[L1], Int),
      axis2: (Axis[L2], Int)
  ): Tensor2[L1, L2] =
    zeros(Shape(axis1, axis2), DType.Float32)

  inline def zeros[L1 <: Label, L2 <: Label, L3 <: Label](
      axis1: (Axis[L1], Int),
      axis2: (Axis[L2], Int),
      axis3: (Axis[L3], Int)
  ): Tensor3[L1, L2, L3] =
    zeros(Shape(axis1, axis2, axis3), DType.Float32)

  def ones[T <: Tuple](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    val jaxValues = Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
    new Tensor[T](jaxValues, shape.labels)

  // Convenient Axis-based overloads for ones
  inline def ones[L1 <: Label](axis1: (Axis[L1], Int)): Tensor1[L1] =
    ones(Shape(axis1), DType.Float32)

  inline def ones[L1 <: Label, L2 <: Label](
      axis1: (Axis[L1], Int),
      axis2: (Axis[L2], Int)
  ): Tensor2[L1, L2] =
    ones(Shape(axis1, axis2), DType.Float32)

  inline def ones[L1 <: Label, L2 <: Label, L3 <: Label](
      axis1: (Axis[L1], Int),
      axis2: (Axis[L2], Int),
      axis3: (Axis[L3], Int)
  ): Tensor3[L1, L2, L3] =
    ones(Shape(axis1, axis2, axis3), DType.Float32)

  /** stack a sequence of tensors along a new axis
    */
  def stack[T <: Tuple, NewAxis <: Label](
      axis: Axis[NewAxis]
  )(
      tensors: Seq[Tensor[T]]
  ): Tensor[Tuple.Concat[Tuple1[NewAxis], T]] =
    require(tensors.nonEmpty, "Cannot stack empty sequence of tensors")
    val refShape = tensors.head.shape
    require(tensors.forall(_.shape.dimensions == refShape.dimensions), "All tensors must have the same shape to stack")
    new Tensor(
      Jax.jnp.stack(tensors.map(_.jaxValue).toPythonProxy), 
      axis.name :: refShape.labels
    )

  /** Concat tensors along an existing axis
    */
  inline def concat[T <: Tuple, ConcatAxis <: Label](
      axis: Axis[ConcatAxis]
  )(
      tensors: Seq[Tensor[T]]
  ): Tensor[T] =
    require(tensors.nonEmpty, "Cannot concat empty sequence of tensors")
    val refAxes = tensors.head.shape.labels
    Tensor(
      Jax.jnp.concatenate(
        tensors.map(_.jaxValue).toPythonProxy, 
        axis = TupleHelpers.indexOf[ConcatAxis, T]
      ),
      axis.name :: refAxes
    )

object Tensor0:
  import Tensor.{Tensor0, Tensor1}

  def apply(jaxValue: Jax.PyDynamic): Tensor0 = Tensor(jaxValue, Nil)

  def apply(value: Float | Int | Boolean): Tensor[EmptyTuple] =
    value match
      case v: Float   => new Tensor[EmptyTuple](Jax.jnp.array(v, dtype=DType.Float32.jaxType), Nil)
      case v: Int     => new Tensor[EmptyTuple](Jax.jnp.array(v, dtype=DType.Int32.jaxType), Nil)
      case v: Boolean => new Tensor[EmptyTuple](Jax.jnp.array(v, dtype=DType.Bool.jaxType), Nil)

object Tensor1:
  import Tensor.{Tensor1, Tensor2}

  def apply[L <: Label](axis: Axis[L], values: ArraySeq[Float], dtype: DType = DType.Float32): Tensor[Tuple1[L]] =
    Tensor(
      Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType), 
      List(axis.name)
    )

  def fromInts[L <: Label](axis: Axis[L], values: ArraySeq[Int], dtype: DType = DType.Int32): Tensor[Tuple1[L]] =
    Tensor(
      Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType), 
      List(axis.name)
    )

object Tensor2:

  import Tensor.{Tensor1, Tensor2}
  import Shape.Shape2

  def apply[L1 <: Label, L2 <: Label](
      shape: Shape2[L1, L2],
      values: ArraySeq[Float],
      dtype: DType,
  ): Tensor[(L1, L2)] = Tensor(shape, values, dtype)

  def apply[L1 <: Label, L2 <: Label](
      shape: Shape2[L1, L2],
      values: ArraySeq[Float],
  ): Tensor[(L1, L2)] = Tensor2(shape, values, DType.Float32)

  def apply[L1 <: Label, L2 <: Label](
      axis1: Axis[L1],
      axis2: Axis[L2],
      values: ArraySeq[ArraySeq[Float]],
      dtype: DType = DType.Float32,
  ): Tensor[(L1, L2)] =
    val rows = values.length
    val cols = values.headOption.map(_.length).getOrElse(0)
    require(values.forall(_.length == cols), "All rows must have the same length")
    Tensor2(Shape(axis1 -> rows, axis2 -> cols), values.flatten, dtype)

  def eye[L <: Label](axis: Axis[L])(dim: Int, dtype: DType = DType.Float32): Tensor2[L, L] = 
    Tensor(
      Jax.jnp.eye(dim, dtype = dtype.jaxType), 
      List(axis.name, axis.name),
    )

  def diag[L <: Label](diag: Tensor1[L]): Tensor2[L, L] =
    Tensor(
      Jax.jnp.diag(diag.jaxValue), 
      diag.shape.labels ++ diag.shape.labels
    )

object Tensor3:

  import Tensor.Tensor3
  import Shape.Shape3

  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      shape: Shape3[L1, L2, L3],
      values: ArraySeq[Float],
      dtype: DType,
  ): Tensor[(L1, L2, L3)] = Tensor(shape, values, dtype)

  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      shape: Shape3[L1, L2, L3],
      values: ArraySeq[Float],
  ): Tensor[(L1, L2, L3)] = Tensor3(shape, values, DType.Float32)

  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      axis1: Axis[L1],
      axis2: Axis[L2],
      axis3: Axis[L3],
      values: ArraySeq[ArraySeq[ArraySeq[Float]]],
      dtype: DType = DType.Float32,
  ): Tensor[(L1, L2, L3)] =
    val dim1 = values.length
    val dim2 = values.headOption.map(_.length).getOrElse(0)
    val dim3 = values.headOption.flatMap(_.headOption).map(_.length).getOrElse(0)
    require(values.forall(_.length == dim2), "All second dimensions must match")
    require(values.forall(_.forall(_.length == dim3)), "All third dimensions must match")
    Tensor3.apply(
      Shape(axis1 -> dim1, axis2 -> dim2, axis3 -> dim3),
      values.flatten.flatten,
      dtype,
    )

class TensorIndexer[T <: Tuple](
    private val tensor: Tensor[T],
    private val index: IntTuple[T]
):

  import Tensor.Tensor0

  val idxAsSeq: Seq[Int] = index.productIterator.toSeq.asInstanceOf[Seq[Int]]

  def get: Tensor0 =
    val indexTuple = Jax.Dynamic.global.tuple(idxAsSeq.toPythonProxy)
    val atHelper = tensor.jaxValue.at.__getitem__(indexTuple)
    val jaxScalar = atHelper.get()
    new Tensor[EmptyTuple](jaxScalar, Nil)

  def set(value: Tensor0): Tensor[T] =
    val jaxValueToSet = value.jaxValue
    val indexTuple = Jax.Dynamic.global.tuple(idxAsSeq.toPythonProxy)
    val atHelper = tensor.jaxValue.at.__getitem__(indexTuple)
    val updatedJaxValue = atHelper.set(jaxValueToSet)
    new Tensor(updatedJaxValue, tensor.axes)
