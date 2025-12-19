package shapeful.tensor

import shapeful.*
import shapeful.Conversions.given
import org.scalacheck.Prop._
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import TensorGen.*
import TestUtil.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class TensorOpsBinarySuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:
  
  py.exec("import jax.numpy as jnp")
      
  def checkBinaryOps[T <: Tuple : Labels](gen: Gen[(Tensor[T], Tensor[T])], suffix: String)(pyCode: String, scOp: (Tensor[T], Tensor[T]) => Tensor[T]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) => 
        val (py, sc) = pythonScalaBinaryOps(t1, t2)(pyCode, scOp)
        py should approxEqual(sc)

  checkBinaryOps(twoTensor0Gen, "+")("t1 + t2", _ + _)
  checkBinaryOps(twoTensor1Gen, "+")("t1 + t2", _ + _)
  checkBinaryOps(twoTensor2Gen, "+")("t1 + t2", _ + _)
  checkBinaryOps(twoTensor3Gen, "+")("t1 + t2", _ + _)

  checkBinaryOps(twoTensor0Gen, "-")("t1 - t2", _ - _)
  checkBinaryOps(twoTensor1Gen, "-")("t1 - t2", _ - _)
  checkBinaryOps(twoTensor2Gen, "-")("t1 - t2", _ - _)
  checkBinaryOps(twoTensor3Gen, "-")("t1 - t2", _ - _)

  checkBinaryOps(twoTensor0Gen, "*")("t1 * t2", _ * _)
  checkBinaryOps(twoTensor1Gen, "*")("t1 * t2", _ * _)
  checkBinaryOps(twoTensor2Gen, "*")("t1 * t2", _ * _)
  checkBinaryOps(twoTensor3Gen, "*")("t1 * t2", _ * _)

  checkBinaryOps(twoTensor0Gen, "/")("t1 / t2", _ / _)
  checkBinaryOps(twoTensor1Gen, "/")("t1 / t2", _ / _)
  checkBinaryOps(twoTensor2Gen, "/")("t1 / t2", _ / _)
  checkBinaryOps(twoTensor3Gen, "/")("t1 / t2", _ / _)

  checkBinaryOps(twoTensor0Gen, "<")("t1 < t2", _ < _)
  checkBinaryOps(twoTensor1Gen, "<")("t1 < t2", _ < _)
  checkBinaryOps(twoTensor2Gen, "<")("t1 < t2", _ < _)
  checkBinaryOps(twoTensor3Gen, "<")("t1 < t2", _ < _)

  checkBinaryOps(twoTensor0Gen, "<=")("t1 <= t2", _ <= _)
  checkBinaryOps(twoTensor1Gen, "<=")("t1 <= t2", _ <= _)
  checkBinaryOps(twoTensor2Gen, "<=")("t1 <= t2", _ <= _)
  checkBinaryOps(twoTensor3Gen, "<=")("t1 <= t2", _ <= _)

  checkBinaryOps(twoTensor0Gen, ">")("t1 > t2", _ > _)
  checkBinaryOps(twoTensor1Gen, ">")("t1 > t2", _ > _)
  checkBinaryOps(twoTensor2Gen, ">")("t1 > t2", _ > _)
  checkBinaryOps(twoTensor3Gen, ">")("t1 > t2", _ > _)

  checkBinaryOps(twoTensor0Gen, ">=")("t1 >= t2", _ >= _)
  checkBinaryOps(twoTensor1Gen, ">=")("t1 >= t2", _ >= _)
  checkBinaryOps(twoTensor2Gen, ">=")("t1 >= t2", _ >= _)
  checkBinaryOps(twoTensor3Gen, ">=")("t1 >= t2", _ >= _)

  checkBinaryOps(twoTensor0Gen, "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoSameTensor0Gen, "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoTensor1Gen, "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoSameTensor1Gen, "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoTensor2Gen, "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoSameTensor2Gen, "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoTensor3Gen, "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinaryOps(twoSameTensor3Gen, "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)

  private def pythonScalaBinaryOps[T <: Tuple : Labels](t1: Tensor[T], t2: Tensor[T])(
    pythonProgram: String,
    scalaProgram: (Tensor[T], Tensor[T]) => Tensor[T],
  ): (Tensor[T], Tensor[T]) =
    require(t1.shape == t2.shape, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")
    val pyRes = {
      py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
      py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
      py.exec(s"res = $pythonProgram")
      Tensor(
        t1.shape, 
        py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
      )
    }
    val scalaRes = scalaProgram(t1, t2)
    (pyRes, scalaRes)
