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

class TensorOpsElementwiseSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:
  
  py.exec("import jax.numpy as jnp")
      
  type A = "a"
  type B = "b"
  type C = "c"

  def check[T <: Tuple : Labels](gen: Gen[Tensor[T]], suffix: String)(pyCode: String, scOp: Tensor[T] => Tensor[T]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t => 
        val (py, sc) = pythonScalaElementwiseOp(t)(pyCode, scOp)
        py should approxEqual(sc)

  check(tensor0Gen, "abs")("jnp.abs(t)", _.abs)
  check(tensor1Gen, "abs")("jnp.abs(t)", _.abs)
  check(tensor2Gen, "abs")("jnp.abs(t)", _.abs)
  check(tensor3Gen, "abs")("jnp.abs(t)", _.abs)

  check(tensor0Gen, "sign")("jnp.sign(t)", _.sign)
  check(tensor1Gen, "sign")("jnp.sign(t)", _.sign)
  check(tensor2Gen, "sign")("jnp.sign(t)", _.sign)
  check(tensor3Gen, "sign")("jnp.sign(t)", _.sign)

  check(tensor0GenOf(min=0, max=100), "sqrt")("jnp.sqrt(t)", _.sqrt)
  check(tensor1GenOf(min=0, max=100), "sqrt")("jnp.sqrt(t)", _.sqrt)
  check(tensor2GenOf(min=0, max=100), "sqrt")("jnp.sqrt(t)", _.sqrt)
  check(tensor3GenOf(min=0, max=100), "sqrt")("jnp.sqrt(t)", _.sqrt)

  check(tensor0GenOf(min=0.1, max=100), "log")("jnp.log(t)", _.log)
  check(tensor1GenOf(min=0.1, max=100), "log")("jnp.log(t)", _.log)
  check(tensor2GenOf(min=0.1, max=100), "log")("jnp.log(t)", _.log)
  check(tensor3GenOf(min=0.1, max=100), "log")("jnp.log(t)", _.log)

  check(tensor0Gen, "sin")("jnp.sin(t)", _.sin)
  check(tensor1Gen, "sin")("jnp.sin(t)", _.sin)
  check(tensor2Gen, "sin")("jnp.sin(t)", _.sin)
  check(tensor3Gen, "sin")("jnp.sin(t)", _.sin)

  check(tensor0Gen, "cos")("jnp.cos(t)", _.cos)
  check(tensor1Gen, "cos")("jnp.cos(t)", _.cos)
  check(tensor2Gen, "cos")("jnp.cos(t)", _.cos)
  check(tensor3Gen, "cos")("jnp.cos(t)", _.cos)

  check(tensor0Gen, "tanh")("jnp.tanh(t)", _.tanh)
  check(tensor1Gen, "tanh")("jnp.tanh(t)", _.tanh)
  check(tensor2Gen, "tanh")("jnp.tanh(t)", _.tanh)
  check(tensor3Gen, "tanh")("jnp.tanh(t)", _.tanh)

  check(tensor0Gen, "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
  check(tensor1Gen, "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
  check(tensor2Gen, "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
  check(tensor3Gen, "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))

  check(tensor0Gen, "unary_-")("jnp.negative(t)", t => -t)
  check(tensor1Gen, "unary_-")("jnp.negative(t)", t => -t)
  check(tensor2Gen, "unary_-")("jnp.negative(t)", t => -t)
  check(tensor3Gen, "unary_-")("jnp.negative(t)", t => -t)

  private def pythonScalaElementwiseOp[T <: Tuple : Labels](in: Tensor[T])(
    pythonProgram: String,
    scalaProgram: Tensor[T] => Tensor[T],
  ): (Tensor[T], Tensor[T]) =
    val pyRes = {
      py.eval("globals()").bracketUpdate("t", in.jaxValue)
      py.exec(s"res = $pythonProgram")
      Tensor(
        in.shape, 
        py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
      )
    }
    val scalaRes = scalaProgram(in)
    (pyRes, scalaRes)
