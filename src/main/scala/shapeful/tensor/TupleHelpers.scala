package shapeful.tensor

import scala.compiletime.{error, erasedValue}

/** Contains basic helper functions for working with tuples (mostly on the type level)
  */
object TupleHelpers:

  type MapTo[T <: Tuple, A] = Tuple.Map[T, [ _ ] =>> A]
  type StringTuple[T <: Tuple] = MapTo[T, String]
  type IntTuple[T <: Tuple] = MapTo[T, Int]

  type Remove[A, B <: Tuple] <: Tuple = B match
    case EmptyTuple      => EmptyTuple
    case A *: EmptyTuple =>
      EmptyTuple // If removing the only element, return empty
    case A *: tail    => tail // If removing first element, return tail
    case head *: tail =>
      head *: Remove[A, tail] // Otherwise, keep head and recurse

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

  /** Helper method to create tuple from sequence (supports up to 6 elements)
    */
  def createTupleFromSeq[T <: Tuple](seq: Seq[Int]): IntTuple[T] =
    require(
      seq.length <= 6,
      s"Tuple size ${seq.length} not supported, maximum is 6"
    )

    seq.length match
      case 0 => EmptyTuple.asInstanceOf[IntTuple[T]]
      case 1 => Tuple1(seq(0)).asInstanceOf[IntTuple[T]]
      case 2 => (seq(0), seq(1)).asInstanceOf[IntTuple[T]]
      case 3 => (seq(0), seq(1), seq(2)).asInstanceOf[IntTuple[T]]
      case 4 => (seq(0), seq(1), seq(2), seq(3)).asInstanceOf[IntTuple[T]]
      case 5 =>
        (seq(0), seq(1), seq(2), seq(3), seq(4)).asInstanceOf[IntTuple[T]]
      case 6 =>
        (seq(0), seq(1), seq(2), seq(3), seq(4), seq(5))
          .asInstanceOf[IntTuple[T]]

  /** Helper method to create tuple from sequence (supports up to 6 elements)
    */
  def createTupleFromSeqString[T <: Tuple](seq: Seq[String]): MapTo[T, String] =
    require(
      seq.length <= 6,
      s"Tuple size ${seq.length} not supported, maximum is 6"
    )

    seq.length match
      case 0 => EmptyTuple.asInstanceOf[StringTuple[T]]
      case 1 => Tuple1(seq(0)).asInstanceOf[StringTuple[T]]
      case 2 => (seq(0), seq(1)).asInstanceOf[StringTuple[T]]
      case 3 => (seq(0), seq(1), seq(2)).asInstanceOf[StringTuple[T]]
      case 4 => (seq(0), seq(1), seq(2), seq(3)).asInstanceOf[StringTuple[T]]
      case 5 =>
        (seq(0), seq(1), seq(2), seq(3), seq(4)).asInstanceOf[StringTuple[T]]
      case 6 =>
        (seq(0), seq(1), seq(2), seq(3), seq(4), seq(5))
          .asInstanceOf[StringTuple[T]]

  /** Get the index of the first occurrence of an element A in a tuple B
    */
  inline def indexOf[A, B <: Tuple]: Int =
    inline erasedValue[B] match
      case _: (A *: tail)    => 0
      case _: (head *: tail) => 1 + indexOf[A, tail]
      case _: EmptyTuple     =>
        error("Element not found in tuple")

  /** Get indices of all elements from ToFind in InTuple
    */
  inline def indicesOf[ToFind <: Tuple, InTuple <: Tuple]: Tuple =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => EmptyTuple
      case _: (head *: tail) =>
        indexOf[head, InTuple] *: indicesOf[tail, InTuple]
