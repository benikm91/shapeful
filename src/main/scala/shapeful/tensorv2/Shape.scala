package shapeful.tensorv2

import shapeful.Label
import scala.collection.View.Empty
import scala.annotation.publicInBinary

/** Represents the (typed) Shape of a tensor with runtime labels
  */
final case class Shape[T <: Tuple] @publicInBinary private[tensorv2] (
    val dimensions: List[Int],
    val labels: List[String],
):

  require(dimensions.size == labels.size, "Dimensions and labels must have the same size")
  require(dimensions.forall(_ > 0), "All dimensions must be positive")

  def rank: Int = dimensions.size
  def size: Int = dimensions.foldLeft(1)((acc, d) => acc * d.asInstanceOf[Int])
  inline def dim[D <: Label](axis: Axis[D]): Int = this.dim[D]
  inline def dim[D <: Label]: Int = dimensions(TupleHelpers.indexOf[D, T])

  def *:[U <: Tuple](other: Shape[U]): Shape[Tuple.Concat[U, T]] =
    new Shape(
      other.dimensions ++ dimensions,
      other.labels ++ labels
    )

  override def toString: String =
    labels.zip(dimensions)
      .map((label, dim) => s"$label=$dim")
      .mkString("Shape(", ", ", ")")

  override def equals(other: Any): Boolean = other match
    case s: Shape[?] => dimensions == s.dimensions && labels == s.labels
    case _           => false

  override def hashCode(): Int = dimensions.hashCode() ^ labels.hashCode()

  def ++[U <: Tuple](other: Shape[U]): Shape[Tuple.Concat[U, T]] =
    new Shape(
      other.dimensions ++ dimensions,
      other.labels ++ labels
    )

  def +:[NewAxis <: Label](dim: (Axis[NewAxis], Int)): Shape[NewAxis *: T] =
    new Shape(
      dim._2 :: dimensions,
      dim._1.name :: labels
    )

object Shape:

  def empty: Shape[EmptyTuple] = new Shape(Nil, Nil)

  private inline def getLabel[L <: Label](idx: Int): String =
    scala.compiletime.summonFrom {
      case v: ValueOf[L] => v.value.toString
      case _             => s"dim$idx"
    }
  
  inline def apply[L <: Label](dim: (Axis[L], Int)): Shape[L *: EmptyTuple] =
    val label = getLabel[L](0)
    new Shape(List(dim._2), List(label))

  inline def apply[L1 <: Label, L2 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int)
  ): Shape[L1 *: L2 *: EmptyTuple] =
    val label1 = getLabel[L1](0)
    val label2 = getLabel[L2](1)
    new Shape(List(dim1._2, dim2._2), List(label1, label2))

  inline def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int),
      dim3: (Axis[L3], Int)
  ): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    val label1 = getLabel[L1](0)
    val label2 = getLabel[L2](1)
    val label3 = getLabel[L3](2)
    new Shape(
      List(dim1._2, dim2._2, dim3._2),
      List(label1, label2, label3)
    )

  type Shape0 = Shape[EmptyTuple]
  type Shape1[L <: Label] = Shape[L *: EmptyTuple]
  type Shape2[L1 <: Label, L2 <: Label] = Shape[L1 *: L2 *: EmptyTuple]
  type Shape3[L1 <: Label, L2 <: Label, L3 <: Label] =
    Shape[L1 *: L2 *: L3 *: EmptyTuple]

val Shape0 = Shape.empty

object Shape1:
  def apply[L <: Label](dim: (Axis[L], Int))(using v: ValueOf[L]): Shape[L *: EmptyTuple] =
    Shape(dim)

object Shape2:
  def apply[L1 <: Label, L2 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int)
  )(using v1: ValueOf[L1], v2: ValueOf[L2]): Shape[L1 *: L2 *: EmptyTuple] =
    Shape(dim1, dim2)

object Test:
  val x = Shape2(Axis["A"] -> 3, Axis["B"] -> 4)

object Shape3:
  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int),
      dim3: (Axis[L3], Int)
  )(using v1: ValueOf[L1], v2: ValueOf[L2], v3: ValueOf[L3]): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    Shape(dim1, dim2, dim3)
