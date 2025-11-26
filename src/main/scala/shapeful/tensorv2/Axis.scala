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

sealed trait AxisIndex[AxisLabel, Shape <: Tuple]:
  def value: Int
class AxisIndexImpl[AxisLabel, Shape <: Tuple](val value: Int) extends AxisIndex[AxisLabel, Shape]

object AxisIndex:

  private inline def indexOf[A, B <: Tuple]: Int =
    inline erasedValue[B] match
      case _: (A *: tail)    => 0
      case _: (head *: tail) => 1 + indexOf[A, tail]
      case _: EmptyTuple     =>
        error("Element not found in tuple")

  private inline def indicesOf[ToFind <: Tuple, InTuple <: Tuple]: Tuple =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => EmptyTuple
      case _: (head *: tail) =>
        indexOf[head, InTuple] *: indicesOf[tail, InTuple]

  inline given [L, S <: Tuple]: AxisIndex[L, S] = 
    new AxisIndexImpl[L, S](indexOf[L, S])