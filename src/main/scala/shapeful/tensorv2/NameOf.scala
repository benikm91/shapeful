package shapeful.tensorv2

import shapeful.tensorv2.TupleHelpers.{RemoverAll, Replacer}

trait NameOf[T]:
    def names: List[String]

class NameOfImpl[T](val names: List[String]) extends NameOf[T]

trait NameOfLowPriority:
  given derivedAllRemover[T <: Tuple, ToRemove <: Tuple, O <: Tuple](using
    removerAll: RemoverAll[T, ToRemove] { type Out = O },
    nameOf: NameOf[T],
    toRemoveNameOf: NameOf[ToRemove],
  ): NameOf[O] = 
    val namesToRemove = toRemoveNameOf.names.toSet
    NameOfImpl[O](
      summon[NameOf[T]].names.filterNot(namesToRemove.contains)
    )

object NameOf extends NameOfLowPriority:

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

    given derivedReplacer[T <: Tuple, ToReplace, OutAxis, O <: Tuple](using
      replacer: Replacer[T, ToReplace, OutAxis] { type Out = O },
      nameOf: NameOf[T],
      toReplaceNameOf: ValueOf[ToReplace],
      outAxisValue: ValueOf[OutAxis],
    ): NameOf[O] = 
      val toReplaceNames = List(toReplaceNameOf.value.toString)
      NameOfImpl[O](
        summon[NameOf[T]].names.map{ name =>
          if toReplaceNames.contains(name) then outAxisValue.value.toString else name
        }
      )

    object ForConcat:

      given [T1 <: Tuple, T2 <: Tuple](
        using
        n1: NameOf[T1],
        n2: NameOf[T2],
      ): NameOf[Tuple.Concat[T1, T2]] = new NameOfImpl(n1.names ++ n2.names)
