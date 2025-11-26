package shapeful.tensorv2

import shapeful.Label
import scala.compiletime.{constValue, erasedValue}

object Axis:
  inline def apply[A <: Label]: Axis[A] = 
    new AxisImpl[A](constValue[A].toString)

  extension [A <: Label](axis: Axis[A])
    def name: String = axis.value.toString
    def asString: String = s"Axis[${name}]"

/**
  * Represents an axis with label A.
  * This maps the type-level label to a runtime representation.
  */
sealed trait Axis[A <: Label]:
  def value: String
class AxisImpl[A <: Label](val value: String) extends Axis[A]

sealed trait AxisIndex[AxisLabel, Shape <: Tuple]:
  def value: Int
class AxisIndexImpl[AxisLabel, Shape <: Tuple](val value: Int) extends AxisIndex[AxisLabel, Shape]

object AxisIndex:
  inline given [L, S <: Tuple]: AxisIndex[L, S] = 
    new AxisIndexImpl[L, S](TupleHelpers.indexOf[L, S])