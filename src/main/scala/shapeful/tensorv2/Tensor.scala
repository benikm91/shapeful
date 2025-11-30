package shapeful.tensorv2

import scala.annotation.targetName
import scala.collection.immutable.ArraySeq
import scala.compiletime.{erasedValue, summonFrom}
import shapeful.jaxv2.Jax
import shapeful.jaxv2.JaxDType
import shapeful.jaxv2.Jax.PyDynamic
import shapeful.Label
import shapeful.tensorv2.TupleHelpers.NameOf
import shapeful.random.Random
import me.shadaj.scalapy.py.SeqConverters

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  // case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])
  case Other(name: String) extends Device(Jax.Dynamic.global.none)

object Device:
  val default: Device = Device.CPU
  val values: Seq[Device] = Seq(
    Device.CPU
  )

case class Tensor[T <: Tuple : NameOf] private[tensorv2] (
  val jaxValue: Jax.PyDynamic,
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)

  lazy val device: Device = Device.values.find(
    d => Jax.device_get(jaxValue).equals(d.jaxDevice)
  ).getOrElse(Device.Other(Jax.device_get(jaxValue).name.as[String]))

  def asType(newDType: DType): Tensor[T] = 
    Tensor(jaxValue = Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(newDType)))

  def toDevice(newDevice: Device): Tensor[T] = 
    Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice))

  def reshape[NewT <: Tuple : NameOf](newShape: Shape[NewT]): Tensor[NewT] =
    require(shape.size == newShape.size, "New shape must have the same number of elements")
    Tensor(Jax.jnp.reshape(jaxValue, newShape.dimensions.toPythonProxy))

  def relabel[From <: Label, To <: Label](from: Axis[From], to: Axis[To]): Tensor[TupleHelpers.Replace[T, From, To]] =
    this.asInstanceOf[Tensor[TupleHelpers.Replace[T, From, To]]]

  def at(idx: Tensor.IndicesOf[T]): TensorIndexer[T] =
    new TensorIndexer(this, idx)

  def tensorEquals(other: Tensor[?]): Boolean =
    Jax.jnp.array_equal(this.jaxValue, other.jaxValue).item().as[Boolean]

  def ==(other: Tensor[?]): Boolean = tensorEquals(other)

  def !=(other: Tensor[?]): Boolean = !(this == other)

  def elementEquals[U <: Tuple](other: Tensor[U])(
    using ev: Tuple.Size[T] =:= Tuple.Size[U]
  ): Tensor[T] =
    require(this.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${this.shape.dimensions} vs ${other.shape.dimensions}")
    Tensor(jaxValue = Jax.jnp.equal(this.jaxValue, other.jaxValue))

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

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [ _ ] =>> Int]

  type Tensor0 = Tensor[EmptyTuple]
  type Tensor1[L <: Label] = Tensor[Tuple1[L]]
  type Tensor2[L1 <: Label, L2 <: Label] = Tensor[(L1, L2)]
  type Tensor3[L1 <: Label, L2 <: Label, L3 <: Label] = Tensor[(L1, L2, L3)]
  type Tensor4[L1 <: Label, L2 <: Label, L3 <: Label, L4 <: Label] = Tensor[(L1, L2, L3, L4)]

  def apply[T <: Tuple : NameOf](shape: Shape[T], values: ArraySeq[Float], dtype: DType = DType.Float32, device: Device = Device.default): Tensor[T] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val jaxValues = Jax.jnp
      .array(
        values.toPythonProxy,
        dtype = dtype.jaxType,
        device = device.jaxDevice,
      )
      .reshape(shape.dimensions.toPythonProxy)
    Tensor(jaxValues)

  def zeros[T <: Tuple : NameOf](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    Tensor(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = dtype.jaxType))

  def ones[T <: Tuple : NameOf](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    Tensor(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = dtype.jaxType))

  /** stack a sequence of tensors along a new axis
    */
  def stack[T <: Tuple : NameOf, NewAxis <: Label : ValueOf](
      axis: Axis[NewAxis]
  )(
      tensors: Seq[Tensor[T]]
  ): Tensor[Tuple.Concat[Tuple1[NewAxis], T]] =
    require(tensors.nonEmpty, "Cannot stack empty sequence of tensors")
    val refShape = tensors.head.shape
    require(tensors.forall(_.shape.dimensions == refShape.dimensions), "All tensors must have the same shape to stack")
    new Tensor(
      Jax.jnp.stack(tensors.map(_.jaxValue).toPythonProxy),
    )

  /** Concat tensors along an existing axis
    */
  inline def concat[T <: Tuple : NameOf, ConcatAxis <: Label](
      axis: Axis[ConcatAxis]
  )(
      tensors: Seq[Tensor[T]]
  )(
    using axisIndex: AxisIndex[ConcatAxis, T],
  ): Tensor[T] =
    require(tensors.nonEmpty, "Cannot concat empty sequence of tensors")
    Tensor(
      Jax.jnp.concatenate(
        tensors.map(_.jaxValue).toPythonProxy, 
        axis = axisIndex.value
      ),
    )

object Tensor0:
  import Tensor.{Tensor0, Tensor1}

  def apply(jaxValue: Jax.PyDynamic): Tensor0 = Tensor(jaxValue)

  def apply(value: Float | Int | Boolean): Tensor0 =
    value match
      case v: Float   => Tensor0(Jax.jnp.array(v, dtype=DType.Float32.jaxType))
      case v: Int     => Tensor0(Jax.jnp.array(v, dtype=DType.Int32.jaxType))
      case v: Boolean => Tensor0(Jax.jnp.array(v, dtype=DType.Bool.jaxType))

object Tensor1:
  import Tensor.{Tensor1, Tensor2}

  def apply[L <: Label : ValueOf](axis: Axis[L], values: ArraySeq[Float], dtype: DType = DType.Float32): Tensor1[L] =
    Tensor(Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

  def fromInts[L <: Label : ValueOf](axis: Axis[L], values: ArraySeq[Int], dtype: DType = DType.Int32): Tensor1[L] =
    Tensor(Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

object Tensor2:

  import Tensor.{Tensor1, Tensor2}
  import Shape.Shape2

  def apply[L1 <: Label : ValueOf, L2 <: Label : ValueOf](
      shape: Shape2[L1, L2],
      values: ArraySeq[Float],
      dtype: DType,
  ): Tensor2[L1, L2] = Tensor(shape, values, dtype)

  def apply[L1 <: Label : ValueOf, L2 <: Label : ValueOf](
      shape: Shape2[L1, L2],
      values: ArraySeq[Float],
  ): Tensor[(L1, L2)] = Tensor2(shape, values, DType.Float32)

  def apply[L1 <: Label : ValueOf, L2 <: Label : ValueOf](
      axis1: Axis[L1],
      axis2: Axis[L2],
      values: ArraySeq[ArraySeq[Float]],
      dtype: DType = DType.Float32,
  ): Tensor[(L1, L2)] =
    val rows = values.length
    val cols = values.headOption.map(_.length).getOrElse(0)
    require(values.forall(_.length == cols), "All rows must have the same length")
    Tensor2(Shape(axis1 -> rows, axis2 -> cols), values.flatten, dtype)

  def eye[L <: Label : ValueOf](axis: Axis[L])(dim: Int, dtype: DType = DType.Float32): Tensor2[L, L] = 
    Tensor(Jax.jnp.eye(dim, dtype = dtype.jaxType))

  def diag[L <: Label : ValueOf](diag: Tensor1[L]): Tensor2[L, L] =
    Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  import Tensor.Tensor3
  import Shape.Shape3

  def apply[L1 <: Label : ValueOf, L2 <: Label : ValueOf, L3 <: Label : ValueOf](
      shape: Shape3[L1, L2, L3],
      values: ArraySeq[Float],
      dtype: DType,
  ): Tensor3[L1, L2, L3] = Tensor(shape, values, dtype)

  def apply[L1 <: Label : ValueOf, L2 <: Label : ValueOf, L3 <: Label : ValueOf](
      shape: Shape3[L1, L2, L3],
      values: ArraySeq[Float],
  ): Tensor3[L1, L2, L3] = Tensor3(shape, values, DType.Float32)

  def apply[L1 <: Label : ValueOf, L2 <: Label : ValueOf, L3 <: Label : ValueOf](
      axis1: Axis[L1],
      axis2: Axis[L2],
      axis3: Axis[L3],
      values: ArraySeq[ArraySeq[ArraySeq[Float]]],
      dtype: DType = DType.Float32,
  ): Tensor3[L1, L2, L3] =
    val dim1 = values.length
    val dim2 = values.headOption.map(_.length).getOrElse(0)
    val dim3 = values.headOption.flatMap(_.headOption).map(_.length).getOrElse(0)
    require(values.forall(_.length == dim2), "All second dimensions must match")
    require(values.forall(_.forall(_.length == dim3)), "All third dimensions must match")
    Tensor3(
      Shape(axis1 -> dim1, axis2 -> dim2, axis3 -> dim3),
      values.flatten.flatten,
      dtype,
    )

class TensorIndexer[T <: Tuple : NameOf](
    private val tensor: Tensor[T],
    private val index: Tensor.IndicesOf[T]
):

  import Tensor.Tensor0

  val idxAsSeq: Seq[Int] = index.productIterator.toSeq.asInstanceOf[Seq[Int]]

  def get: Tensor0 =
    val indexTuple = Jax.Dynamic.global.tuple(idxAsSeq.toPythonProxy)
    val atHelper = tensor.jaxValue.at.__getitem__(indexTuple)
    val jaxScalar = atHelper.get()
    new Tensor0(jaxScalar)

  def set(value: Tensor0): Tensor[T] =
    val jaxValueToSet = value.jaxValue
    val indexTuple = Jax.Dynamic.global.tuple(idxAsSeq.toPythonProxy)
    val atHelper = tensor.jaxValue.at.__getitem__(indexTuple)
    val updatedJaxValue = atHelper.set(jaxValueToSet)
    new Tensor(updatedJaxValue)
