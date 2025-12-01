package shapeful.tensorv2

import shapeful.Label
import scala.compiletime.{constValue, erasedValue}

import scala.compiletime.error

object Axis:
  def apply[A <: Label]: Axis[A] = 
    new AxisImpl[A]()

/**
  * TODO Is this Shape1?
  * Represents an axis with label A.
  * This maps the type-level label to a runtime representation.
  */
sealed trait Axis[A <: Label]
class AxisImpl[A <: Label] extends Axis[A]

sealed trait AxisIndex[Shape <: Tuple, AxisLabel]:
  def value: Int

object AxisIndex:
  
  class AxisIndexImpl[Shape <: Tuple, AxisLabel](val value: Int) extends AxisIndex[Shape, AxisLabel]

  private inline def indexOf[ B <: Tuple, ToFind]: Int =
    inline erasedValue[B] match
      case _: (ToFind *: tail)    => 0
      case _: (head *: tail) => 1 + indexOf[tail, ToFind]
      case _: EmptyTuple     =>
        error("Element not found in tuple")

  private inline def indicesOf[InTuple <: Tuple, ToFind <: Tuple]: Tuple =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => EmptyTuple
      case _: (head *: tail) =>
        indexOf[InTuple, head] *: indicesOf[InTuple, tail]

  inline given [T <: Tuple, L]: AxisIndex[T, L] = AxisIndexImpl[T, L](indexOf[T, L])

sealed trait AxisIndices[T <: Tuple, AxisLabels <: Tuple]:
  def values: List[Int]

import scala.compiletime.{constValue, erasedValue, summonInline}
object AxisIndices:

  class AxisIndicesImpl[T <: Tuple, AxisLabels <: Tuple](val values: List[Int]) extends AxisIndices[T, AxisLabels]
  
  private inline def indicesOfList[InTuple <: Tuple, ToFind <: Tuple]: List[Int] =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => Nil
      case _: (head *: tail) =>
        summonInline[AxisIndex[InTuple, head]].value :: indicesOfList[InTuple, tail]

  inline given [T <: Tuple, ToFind <: Tuple]: AxisIndices[T, ToFind] = AxisIndicesImpl[T, ToFind](indicesOfList[T, ToFind])

end AxisIndices