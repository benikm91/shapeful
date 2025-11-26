package shapeful.tensor

import shapeful.Label

object Axis:
  inline def apply[A <: Label]: Axis[A] = new Axis[A]()

  extension [A <: Label](axis: Axis[A])
    def name(using v: ValueOf[A]): String = v.value.toString
    def asString(using v: ValueOf[A]): String = s"Axis[${v.value}]"

/**
  * Represents an axis with label A.
  * This maps the type-level label to a runtime representation.
  */
sealed case class Axis[A <: Label]()
