package shapeful.tensorv2

import scala.compiletime.{error, erasedValue, constValue, summonInline}
import shapeful.Label

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

  type ShapeTreeValue = String

  enum ShapeTree[+A]:
    case Empty extends ShapeTree[Nothing]
    case Leaf(name: A) extends ShapeTree[A]
    case Node(children: List[ShapeTree[A]])extends ShapeTree[A]

    def width: Int = this match 
      case ShapeTree.Empty => 0
      case ShapeTree.Leaf(name) => 1
      case ShapeTree.Node(children) => children.size

    def toList: List[A] = this match
      case Empty => Nil
      case Leaf(name) => List(name)
      case Node(children) => children.flatMap(_.toList)

    def mkString(sep: String, groupStart: String = "", groupEnd: String = "", nextGroupStart: String = "(", nextGroupEnd: String = ")"): String = this match
      case Empty => ""
      case Leaf(name) => name.toString
      case Node(children) => groupStart + children.map(_.mkString(sep, nextGroupStart, nextGroupEnd)).mkString(sep) + groupEnd

    /** Remove the leaf at the given index from the NameTree. */
    def remove(idx: Int): ShapeTree[A] =
      def helper(tree: ShapeTree[A], currentIdx: Int): (ShapeTree[A], Int)
        = tree match
          case Empty => (Empty, currentIdx)
          case Leaf(name) =>
            if currentIdx == idx then (Empty, currentIdx + 1)
            else (Leaf(name), currentIdx + 1)
          case Node(children) =>
            val (newChildren, newIdx) = children.foldLeft((List.empty[ShapeTree[A]], currentIdx)) {
              case ((acc, curIdx), child) =>
                val (newChild, nextIdx) = helper(child, curIdx)
                if newChild == Empty then (acc, nextIdx)
                else (acc :+ newChild, nextIdx)
            }
            (Node(newChildren), newIdx)
      helper(this, 0)._1

    def ::[B >: A](other: ShapeTree[B]): ShapeTree[B] = 
      (this, other) match
        case (ShapeTree.Empty, _) => other
        case (_, ShapeTree.Empty) => this
        case (Leaf(_), Node(lst)) => Node(this :: lst)
        case (Node(lst), Leaf(_)) => Node(lst :+ other)
        case (Node(lst1), Node(lst2)) => Node(lst1 ++ lst2)
        case (Leaf(_), Leaf(_)) => Node(List(this, other))

  trait ShapeTreeOf[T]:
    def tree: ShapeTree[ShapeTreeValue]

  class ShapeTreeOfImpl[T](val tree: ShapeTree[ShapeTreeValue]) extends ShapeTreeOf[T]
  object ShapeTreeOf:

    // empty case
    given namesOfEmpty: ShapeTreeOf[EmptyTuple] =
      new ShapeTreeOfImpl[EmptyTuple](ShapeTree.Empty)

    // lift ValueOf of to NameOf
    given [head](using
        headName: ValueOf[head],
    ): ShapeTreeOf[head] = new ShapeTreeOfImpl[head](ShapeTree.Leaf(headName.value.toString))
    
    // Stack a tuple to group of leaves
    given [A, B](using  a: ShapeTreeOf[A], b: ShapeTreeOf[B]): ShapeTreeOf[(A, B)] = new ShapeTreeOfImpl[(A, B)](ShapeTree.Node(List(a.tree, b.tree)))
    given [A, B, C](using  a: ShapeTreeOf[A], b: ShapeTreeOf[B], c: ShapeTreeOf[C]): ShapeTreeOf[(A, B, C)] = new ShapeTreeOfImpl[(A, B, C)](ShapeTree.Node(List(a.tree, b.tree, c.tree)))
    given [A, B, C, D](using  a: ShapeTreeOf[A], b: ShapeTreeOf[B], c: ShapeTreeOf[C], d: ShapeTreeOf[D]): ShapeTreeOf[(A, B, C, D)] = new ShapeTreeOfImpl[(A, B, C, D)](ShapeTree.Node(List(a.tree, b.tree, c.tree, d.tree)))
    given [A, B, C, D, E](using  a: ShapeTreeOf[A], b: ShapeTreeOf[B], c: ShapeTreeOf[C], d: ShapeTreeOf[D], e: ShapeTreeOf[E]): ShapeTreeOf[(A, B, C, D, E)] = new ShapeTreeOfImpl[(A, B, C, D, E)](ShapeTree.Node(List(a.tree, b.tree, c.tree, d.tree, e.tree)))
    given [A, B, C, D, E, F](using  a: ShapeTreeOf[A], b: ShapeTreeOf[B], c: ShapeTreeOf[C], d: ShapeTreeOf[D], e: ShapeTreeOf[E], f: ShapeTreeOf[F]): ShapeTreeOf[(A, B, C, D, E, F)] = new ShapeTreeOfImpl[(A, B, C, D, E, F)](ShapeTree.Node(List(a.tree, b.tree, c.tree, d.tree, e.tree, f.tree)))

    // append a value to a tuple
    given [head, tail <: Tuple](
      using 
      headName: ValueOf[head],
      tailNames: ShapeTreeOf[tail]
    ): ShapeTreeOf[head *: tail] = new ShapeTreeOfImpl[head *: tail](
      ShapeTree.Leaf(headName.value.toString) :: tailNames.tree
    )

    object ForConcat:
      given concatNames[A <: Tuple, B <: Tuple](using
        namesA: ShapeTreeOf[A],
        namesB: ShapeTreeOf[B]
      ): ShapeTreeOf[Tuple.Concat[A, B]] =
        new ShapeTreeOfImpl[Tuple.Concat[A, B]](namesA.tree :: namesB.tree)

    object ForRemove:
      given derivedRemoveNames[A, T <: Tuple](using 
        base: ShapeTreeOf[T], 
        idx: AxisIndex[A, T]
      ): ShapeTreeOf[TupleHelpers.Remove[A, T]] = 
        new ShapeTreeOfImpl(base.tree.remove(idx.value))

      given derivedRemoveTwoNames[A, B, From <: Tuple](using
        base: ShapeTreeOf[From],
        idx1: AxisIndex[A, From],
        idx2: AxisIndex[B, From],
      ): ShapeTreeOf[TupleHelpers.Remove[A, TupleHelpers.Remove[B, From]]] =
        new ShapeTreeOfImpl(
          base.tree
            .remove(idx2.value)
            .remove(if idx1.value < idx2.value then idx1.value else idx1.value - 1)
        )

      given derivedRemoveThreeNames[A, B, C, From <: Tuple](using
        base: ShapeTreeOf[From],
        idx1: AxisIndex[A, From],
        idx2: AxisIndex[B, From],
        idx3: AxisIndex[C, From],
      ): ShapeTreeOf[TupleHelpers.Remove[A, TupleHelpers.Remove[B, TupleHelpers.Remove[C, From]]]] =
        new ShapeTreeOfImpl(
          base.tree
            .remove(idx3.value)
            .remove(if idx2.value < idx3.value then idx2.value else idx2.value - 1)
            .remove(
              if idx1.value < idx2.value && idx1.value < idx3.value then idx1.value
              else if (idx1.value > idx2.value && idx1.value < idx3.value) || (idx1.value < idx2.value && idx1.value > idx3.value) then idx1.value - 1
              else idx1.value - 2
            )
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