package shapeful.tensor

import shapeful.*
import shapeful.Conversions.given
import org.scalacheck.Prop._
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import TensorGen.*
import org.scalacheck.Prop.forAll

import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import org.scalatest.matchers.{Matcher, MatchResult}

object TestUtil:

    def approxEqual[T <: Tuple : Labels](right: Tensor[T]): Matcher[Tensor[T]] = 
        new Matcher[Tensor[T]]:
            def apply(left: Tensor[T]): MatchResult =
                val areEqual = left `approxEquals` right
                
                lazy val diffMsg = if areEqual then "" else s"Max diff: ${(left - right).abs.max}"

                MatchResult(
                    areEqual,
                    s"Tensors did not match ($diffMsg).\nLeft (Py): $left\nRight (Sc): $right",
                    s"Tensors matched, but they shouldn't have."
                )

