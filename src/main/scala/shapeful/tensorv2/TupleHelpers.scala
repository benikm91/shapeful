package shapeful.tensorv2

import scala.compiletime.{error, erasedValue, constValue}

/** Contains basic helper functions for working with tuples (mostly on the type level)
  */
object TupleHelpers:

  type MapTo[T <: Tuple, A] = Tuple.Map[T, [ _ ] =>> A]
  opaque type StringTuple[T <: Tuple] = MapTo[T, String]
  
  type Remove[A, B <: Tuple] <: Tuple = B match
    case EmptyTuple       => EmptyTuple
    case A *: EmptyTuple  => EmptyTuple
    case A *: tail        => tail
    case head *: tail     => head *: Remove[A, tail]

  /** Remove all occurrences of elements in ToRemove from From
    */
  type RemoveAll[ToRemove <: Tuple, From <: Tuple] <: Tuple = ToRemove match
    case EmptyTuple   => From
    case head *: tail => RemoveAll[tail, Remove[head, From]]

  /** Compute the result shape after contracting over a single axis Result is concatenation of T1 and T2 with
    * ContractAxis removed from both
    */
  type ContractResult[T1 <: Tuple, T2 <: Tuple, ContractAxis] =
    Tuple.Concat[Remove[ContractAxis, T1], Remove[ContractAxis, T2]]

  type Replace[T <: Tuple, Needle, Replacement] = Tuple.Map[T, [ A ] =>>
    A match
      case Needle => Replacement
      case _      => A
  ]

  trait NamesOf[T <: Tuple]:
    def value: List[String]

  class NamesOfImpl[T <: Tuple](val value: List[String]) extends NamesOf[T]

  object NamesOf:
    inline given namesOfEmpty: NamesOf[EmptyTuple] =
      new NamesOfImpl[EmptyTuple](Nil)

    inline given [head, tail <: Tuple](using
        headName: ValueOf[head],
        tailNames: NamesOf[tail]
    ): NamesOf[head *: tail] =
      new NamesOfImpl[head *: tail](headName.value.toString :: tailNames.value)

  inline def namesOf[T <: Tuple]: List[String] =
    inline erasedValue[T] match
      case _: (head *: tail) => constValue[head].toString :: namesOf[tail]
      case _: EmptyTuple     => Nil

  inline def indexOf[A, B <: Tuple]: Int =
    inline erasedValue[B] match
      case _: (A *: tail)    => 0
      case _: (head *: tail) => 1 + indexOf[A, tail]
      case _: EmptyTuple     =>
        error("Element not found in tuple")

  inline def indicesOf[ToFind <: Tuple, InTuple <: Tuple]: Tuple =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => EmptyTuple
      case _: (head *: tail) =>
        indexOf[head, InTuple] *: indicesOf[tail, InTuple]
