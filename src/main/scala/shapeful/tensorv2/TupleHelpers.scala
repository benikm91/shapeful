package shapeful.tensorv2

import scala.compiletime.{error, erasedValue, constValue, summonInline}
import shapeful.Label
import scala.util.NotGiven
import shapeful.tensorv2.TupleHelpers.ValuesOf.WrapAxes

import scala.util.NotGiven
import shapeful.tensorv2.TupleHelpers.UnwrapAxes
import javax.smartcardio.ATR

/***
 * Type class to remove an axis from a tuple of axes.
 */
trait Remover[Axes <: Tuple, Axis]:
  type Out <: Tuple

object Remover:
  
  given headMatch[A, Tail <: Tuple]: Remover[A *: Tail, A] with
    type Out = Tail

  // recurse[L3, L2, L3 *: EmptyTuple.type]
  given recurse[A, H, T <: Tuple, TailOut <: Tuple](
    using 
    next: Remover[T, A] { type Out = TailOut },
    ev: Tuple.Contains[T, A] =:= true,
  ): Remover[H *: T, A] with
    type Out = H *: next.Out

trait RemoverAll[T <: Tuple, ToRemove <: Tuple]:
  type Out <: Tuple

object RemoverAll:

  given empty[T <: Tuple]: RemoverAll[T, EmptyTuple] with
    type Out = T

  given recurse[Head, TailToRemove <: Tuple, From <: Tuple, Intermediate <: Tuple, OutTail <: Tuple](
    using
    remover: Remover[From, Head] { type Out = Intermediate },
    next: RemoverAll[Intermediate, TailToRemove] { type Out = OutTail }
  ): RemoverAll[From, Head *: TailToRemove] with
    type Out = OutTail

type AxesTuple[T <: Tuple] <: Tuple = T match 
  case EmptyTuple       => EmptyTuple
  case head *: tail     => Axis[head] *: AxesTuple[tail]

type UnAxesTuple[T <: Tuple] <: Tuple = T match 
  case EmptyTuple       => EmptyTuple
  case AxesTuple[t]     => t

trait AxesUnwrapper[AT <: Tuple]:
  type Out <: Tuple

object AxesUnwrapper:

  given empty: AxesUnwrapper[EmptyTuple] with
    type Out = EmptyTuple

  given recurse[H <: Label, AT <: Tuple](
    using
    next: AxesUnwrapper[AT] { 
      type Out = UnwrapAxes[AT] 
    }
  ): AxesUnwrapper[Axis[H] *: AT] with
    type Out = H *: next.Out

trait AxesWrapper[T <: Tuple, AT <: AxesTuple[T]]:
  type Out <: AT

object AxesWrapper:
  given empty: AxesWrapper[EmptyTuple, AxesTuple[EmptyTuple]] with
    type Out = EmptyTuple
  given recurse[H <: Label, T <: Tuple, AT <: AxesTuple[T]](
    using
    next: AxesWrapper[T, AT] { type Out = AT }
  ): AxesWrapper[H *: T, Axis[H] *: AT] with
    type Out = Axis[H] *: AT

/** Contains basic helper functions for working with tuples (mostly on the type level)
  */
object TupleHelpers:

  type MapTo[T <: Tuple, A] = Tuple.Map[T, [ _ ] =>> A]
  opaque type StringTuple[T <: Tuple] = MapTo[T, String]
  
  type Remove[T <: Tuple, A] <: Tuple = T match
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
    Tuple.Concat[Remove[T1, ContractAxis], Remove[T2, ContractAxis]]

  type Replace[T <: Tuple, Needle, Replacement] = Tuple.Map[T, [ A ] =>>
    A match
      case Needle => Replacement
      case _      => A
  ]

  trait NameOf[T]:
    def names: List[String]

  class NameOfImpl[T](val names: List[String]) extends NameOf[T]

  object NameOf:

    given namesOfEmpty: NameOf[EmptyTuple] = new NameOfImpl[EmptyTuple](Nil)

    given lift[A] (using v: ValueOf[A]): NameOf[A] = new NameOfImpl[A](List(v.value.toString))

    given [A, B](using  a: NameOf[A], b: NameOf[B]): NameOf[(A, B)] = new NameOfImpl[(A, B)](a.names ++ b.names)
    given [A, B, C](using  a: NameOf[A], b: NameOf[B], c: NameOf[C]): NameOf[(A, B, C)] = new NameOfImpl[(A, B, C)](a.names ++ b.names ++ c.names)
    given [A, B, C, D](using  a: NameOf[A], b: NameOf[B], c: NameOf[C], d: NameOf[D]): NameOf[(A, B, C, D)] = new NameOfImpl[(A, B, C, D)](a.names ++ b.names ++ c.names ++ d.names)
    given [A, B, C, D, E](using  a: NameOf[A], b: NameOf[B], c: NameOf[C], d: NameOf[D], e: NameOf[E]): NameOf[(A, B, C, D, E)] = new NameOfImpl[(A, B, C, D, E)](a.names ++ b.names ++ c.names ++ d.names ++ e.names)
    given [A, B, C, D, E, F](using  a: NameOf[A], b: NameOf[B], c: NameOf[C], d: NameOf[D], e: NameOf[E], f: NameOf[F]): NameOf[(A, B, C, D, E, F)] = new NameOfImpl[(A, B, C, D, E, F)](a.names ++ b.names ++ c.names ++ d.names ++ e.names ++ f.names)  
    
    given [head, tail <: Tuple](
      using 
      v: ValueOf[head],
      t: NameOf[tail],
    ): NameOf[head *: tail] = new NameOfImpl[head *: tail](
      v.value.toString :: t.names
    )

    def removerNameOf[T <: Tuple : NameOf, A : ValueOf](
      remover: Remover[T, A],
    ): NameOf[remover.Out] = NameOfImpl[remover.Out](
      summon[NameOf[T]].names.filterNot(_ == summon[ValueOf[A]].value.toString)
    )

    def removerAllNameOf[T <: Tuple : NameOf, ToRemove <: Tuple : NameOf](
      remover: RemoverAll[T, ToRemove],
    ): NameOf[remover.Out] = 
      val namesToRemove = summon[NameOf[ToRemove]].names.toSet
      NameOfImpl[remover.Out](
        summon[NameOf[T]].names.filterNot(namesToRemove.contains)
      )

    object ForConcat:

      given [T1 <: Tuple, T2 <: Tuple](
        using
        n1: NameOf[T1],
        n2: NameOf[T2],
      ): NameOf[Tuple.Concat[T1, T2]] = new NameOfImpl(n1.names ++ n2.names)

    object ForRemoveAll:
      import scala.annotation.tailrec

      @tailrec
      private def removeAllNames(names: List[String], indices: List[Int], offset: Int = 0): List[String] =
        indices match
          case Nil => names
          case head :: tail =>
            val adjustedIndex = head - offset
            removeAllNames(names.patch(adjustedIndex, Nil, 1), tail, offset + 1)

      given derivedRemoveAllNames[ToRemove <: Tuple, From <: Tuple](using
        base: NameOf[From],
        indices: AxisIndices[ToRemove, From],
      ): NameOf[TupleHelpers.RemoveAll[ToRemove, From]] =
        new NameOfImpl(removeAllNames(base.names, indices.values.sorted))

  type UnwrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple => EmptyTuple
    case Axis[a] *: tail => a *: UnwrapAxes[tail]
    case h *: tail => h *: UnwrapAxes[tail]
  
  trait DimExtractor[T]:
    def extract(t: T): Map[String, Int]

  object DimExtractor:
    given DimExtractor[EmptyTuple] with
      def extract(t: EmptyTuple) = Map.empty

    given [L <: Label, Tail <: Tuple](using
      labelValue: ValueOf[L],
      tailExtractor: DimExtractor[Tail]
    ): DimExtractor[(Axis[L], Int) *: Tail] with
      def extract(t: (Axis[L], Int) *: Tail) =
        val (_, size) = t.head
        Map(labelValue.value.toString -> size) ++ tailExtractor.extract(t.tail)


  import shapeful.Label
  import scala.compiletime.constValue

  // Create type class ValuesOf for Tuple that ensures ValueOf is available for each element
  trait ValuesOf[T <: Tuple]:
    def values: List[String]
  object ValuesOf:
    given ValuesOf[EmptyTuple] with
      def values = Nil
    given [H <: Label : ValueOf, Tail <: Tuple](using tailValues: ValuesOf[Tail]): ValuesOf[H *: Tail] with
      def values = summon[ValueOf[H]].value.toString :: tailValues.values
  
    type WrapAxes[T <: Tuple] <: Tuple = T match 
      case EmptyTuple       => EmptyTuple
      case head *: tail     => Axis[head] *: WrapAxes[tail]

    trait AxesFactory[T <: Tuple]:
      def apply(): WrapAxes[T]

    object AxesFactory:
      given axesFactoryEmpty: AxesFactory[EmptyTuple] with
        def apply(): EmptyTuple = EmptyTuple

      given [H <: Label : ValueOf, Tail <: Tuple](using tailFactory: AxesFactory[Tail]): AxesFactory[H *: Tail] with
        def apply(): Axis[H] *: WrapAxes[Tail] = 
          val headAxis = Axis[H] 
          val tailAxes = tailFactory.apply()
          headAxis *: tailAxes
