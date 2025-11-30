package shapeful.tensorv2

import scala.compiletime.{error, erasedValue, constValue, summonInline}
import shapeful.Label
import scala.util.NotGiven

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

  trait NameOf[T]:
    def tree: List[String]

  class NameOfImpl[T](val tree: List[String]) extends NameOf[T]

  object NameOf:

    // empty case
    given namesOfEmpty: NameOf[EmptyTuple] =
      new NameOfImpl[EmptyTuple](Nil)

    // lift ValueOf of to NameOf
    given [head] (using
        v: ValueOf[head],
    ): NameOf[head] = new NameOfImpl[head](List(v.value.toString))

    // Stack a tuple to group of leaves
    given [A, B](using  a: NameOf[A], b: NameOf[B]): NameOf[(A, B)] = new NameOfImpl[(A, B)](a.tree ++ b.tree)
    given [A, B, C](using  a: NameOf[A], b: NameOf[B], c: NameOf[C]): NameOf[(A, B, C)] = new NameOfImpl[(A, B, C)](a.tree ++ b.tree ++ c.tree)
    given [A, B, C, D](using  a: NameOf[A], b: NameOf[B], c: NameOf[C], d: NameOf[D]): NameOf[(A, B, C, D)] = new NameOfImpl[(A, B, C, D)](a.tree ++ b.tree ++ c.tree ++ d.tree)
    given [A, B, C, D, E](using  a: NameOf[A], b: NameOf[B], c: NameOf[C], d: NameOf[D], e: NameOf[E]): NameOf[(A, B, C, D, E)] = new NameOfImpl[(A, B, C, D, E)](a.tree ++ b.tree ++ c.tree ++ d.tree ++ e.tree)
    given [A, B, C, D, E, F](using  a: NameOf[A], b: NameOf[B], c: NameOf[C], d: NameOf[D], e: NameOf[E], f: NameOf[F]): NameOf[(A, B, C, D, E, F)] = new NameOfImpl[(A, B, C, D, E, F)](a.tree ++ b.tree ++ c.tree ++ d.tree ++ e.tree ++ f.tree)  
    
    // append a value to a tuple
    given [head, tail <: Tuple](
      using 
      v: ValueOf[head],
      t: NameOf[tail],
    ): NameOf[head *: tail] = new NameOfImpl[head *: tail](
      v.value.toString :: t.tree
    )

    object ForConcat:
      given concatNames[V, A <: Tuple, B <: Tuple](using
        namesA: NameOf[A],
        namesB: NameOf[B]
      ): NameOf[Tuple.Concat[A, B]] =
        new NameOfImpl[Tuple.Concat[A, B]](namesA.tree ++ namesB.tree)

    object ForRemove:
      given derivedRemoveNames[V, A, T <: Tuple](using 
        base: NameOf[T], 
        idx: AxisIndex[A, T]
      ): NameOf[TupleHelpers.Remove[A, T]] = 
        new NameOfImpl(base.tree.patch(idx.value, Nil, 1))

      given derivedRemoveTwoNames[V, A, B, From <: Tuple](using
        base: NameOf[From],
        idx1: AxisIndex[A, From],
        idx2: AxisIndex[B, From],
      ): NameOf[TupleHelpers.Remove[A, TupleHelpers.Remove[B, From]]] =
        new NameOfImpl(
          base.tree
            .patch(idx2.value, Nil, 1)
            .patch(if idx1.value < idx2.value then idx1.value else idx1.value - 1, Nil, 1)
        )

      given derivedRemoveThreeNames[V, A, B, C, From <: Tuple](using
        base: NameOf[From],
        idx1: AxisIndex[A, From],
        idx2: AxisIndex[B, From],
        idx3: AxisIndex[C, From],
      ): NameOf[TupleHelpers.Remove[A, TupleHelpers.Remove[B, TupleHelpers.Remove[C, From]]]] =
        new NameOfImpl(
          base.tree
            .patch(idx3.value, Nil, 1)
            .patch(if idx2.value < idx3.value then idx2.value else idx2.value - 1, Nil, 1)
            .patch(
              if idx1.value < idx2.value && idx1.value < idx3.value then idx1.value
              else if (idx1.value > idx2.value && idx1.value < idx3.value) || (idx1.value < idx2.value && idx1.value > idx3.value) then idx1.value - 1
              else idx1.value - 2
            , Nil, 1)
        )

    object ForContractResult:
      export ForConcat.concatNames
      export ForRemove.derivedRemoveNames

  type TupleFlat[T <: Tuple] <: Tuple = T match
    case EmptyTuple => EmptyTuple
    case (h *: t) *: tail => 
      Tuple.Concat[TupleFlat[h *: t], TupleFlat[tail]]
    case EmptyTuple *: tail => 
      TupleFlat[tail]
    case h *: tail => 
      h *: TupleFlat[tail]

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