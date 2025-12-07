package shapeful.tensorv2

object TupleHelpers:

  type Remover[T <: Tuple, ToRemoveElement] = RemoverAll[T, ToRemoveElement *: EmptyTuple]

  trait RemoverAll[T <: Tuple, ToRemove <: Tuple]:
    type Out <: Tuple

  object RemoverAll:

    given emptyKeys[T <: Tuple]: RemoverAll[T, EmptyTuple] with
      type Out = T

    given chain[T <: Tuple, K1, K2, Rest <: Tuple, Inter <: Tuple, O <: Tuple](using
      r1: RemoverAll[T, K1 *: EmptyTuple] { type Out = Inter },
      r2: RemoverAll[Inter, K2 *: Rest] { type Out = O }
    ): RemoverAll[T, K1 *: K2 *: Rest] with
      type Out = r2.Out

    given singleFound[K, Tail <: Tuple]: RemoverAll[K *: Tail, K *: EmptyTuple] with
      type Out = Tail

    given singleSearch[H, Tail <: Tuple, K, TailOut <: Tuple](using
      next: RemoverAll[Tail, K *: EmptyTuple] { type Out = TailOut }
    ): RemoverAll[H *: Tail, K *: EmptyTuple] with
      type Out = H *: TailOut

  trait Replacer[T <: Tuple, Target, Replacement]:
    type Out <: Tuple

  object Replacer extends ReplacerLowPriority:

    given found[Target, Tail <: Tuple, Replacement]: Replacer[Target *: Tail, Target, Replacement] with
      type Out = Replacement *: Tail

    given empty[Target, Replacement]: Replacer[EmptyTuple, Target, Replacement] with
      type Out = EmptyTuple

  trait ReplacerLowPriority:
    given recurse[Head, Tail <: Tuple, Target, Replacement, TailOut <: Tuple](using
      next: Replacer[Tail, Target, Replacement] { type Out = TailOut }
    ): Replacer[Head *: Tail, Target, Replacement] with
      type Out = Head *: TailOut