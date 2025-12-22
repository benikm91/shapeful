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

class TensorOpsReductionSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:
  
  py.exec("import jax.numpy as jnp")
      
  def checkBinaryReductionOpsToBool[T <: Tuple : Labels](gen: Gen[(Tensor[T], Tensor[T])], suffix: String)(pyCode: String, scOp: (Tensor[T], Tensor[T]) => Boolean) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) => 
        val (py, sc) = pythonScalaBinaryReductionOpsToBool(t1, t2)(pyCode, scOp)
        py shouldEqual sc

  def checkReductionOpsToFloat[T <: Tuple : Labels](gen: Gen[Tensor[T]], suffix: String)(pyCode: String, scOp: Tensor[T] => Tensor0) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t => 
        val (py, sc) = pythonScalaReductionOpsToFloat(t)(pyCode, scOp)
        py shouldEqual sc

  def checkReductionOpsToBool[T <: Tuple : Labels](gen: Gen[Tensor[T]], suffix: String)(pyCode: String, scOp: Tensor[T] => Boolean) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t => 
        val (py, sc) = pythonScalaReductionOpsToBool(t)(pyCode, scOp)
        py shouldEqual sc

  checkBinaryReductionOpsToBool(twoTensor0Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor0Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoTensor1Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor1Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoTensor2Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor2Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoTensor3Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor3Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)

  checkReductionOpsToFloat(tensor0Gen, "sum")("jnp.sum(t)", _.sum)
  checkReductionOpsToFloat(tensor1Gen, "sum")("jnp.sum(t)", _.sum)
  checkReductionOpsToFloat(tensor2Gen, "sum")("jnp.sum(t)", _.sum)
  checkReductionOpsToFloat(tensor3Gen, "sum")("jnp.sum(t)", _.sum)

  checkReductionOpsToFloat(tensor0Gen, "mean")("jnp.mean(t)", _.mean)
  checkReductionOpsToFloat(tensor1Gen, "mean")("jnp.mean(t)", _.mean)
  checkReductionOpsToFloat(tensor2Gen, "mean")("jnp.mean(t)", _.mean)
  checkReductionOpsToFloat(tensor3Gen, "mean")("jnp.mean(t)", _.mean)

  checkReductionOpsToFloat(tensor0Gen, "std")("jnp.std(t)", _.std)
  checkReductionOpsToFloat(tensor1Gen, "std")("jnp.std(t)", _.std)
  checkReductionOpsToFloat(tensor2Gen, "std")("jnp.std(t)", _.std)
  checkReductionOpsToFloat(tensor3Gen, "std")("jnp.std(t)", _.std)

  checkReductionOpsToFloat(tensor0Gen, "max")("jnp.max(t)", _.max)
  checkReductionOpsToFloat(tensor1Gen, "max")("jnp.max(t)", _.max)
  checkReductionOpsToFloat(tensor2Gen, "max")("jnp.max(t)", _.max)
  checkReductionOpsToFloat(tensor3Gen, "max")("jnp.max(t)", _.max)

  checkReductionOpsToFloat(tensor0Gen, "min")("jnp.min(t)", _.min)
  checkReductionOpsToFloat(tensor1Gen, "min")("jnp.min(t)", _.min)
  checkReductionOpsToFloat(tensor2Gen, "min")("jnp.min(t)", _.min)
  checkReductionOpsToFloat(tensor3Gen, "min")("jnp.min(t)", _.min)

  checkReductionOpsToFloat(tensor0Gen, "argmax")("jnp.argmax(t)", _.argmax)
  checkReductionOpsToFloat(tensor1Gen, "argmax")("jnp.argmax(t)", _.argmax)
  checkReductionOpsToFloat(tensor2Gen, "argmax")("jnp.argmax(t)", _.argmax)
  checkReductionOpsToFloat(tensor3Gen, "argmax")("jnp.argmax(t)", _.argmax)

  checkReductionOpsToFloat(tensor0Gen, "argmin")("jnp.argmin(t)", _.argmin)
  checkReductionOpsToFloat(tensor1Gen, "argmin")("jnp.argmin(t)", _.argmin)
  checkReductionOpsToFloat(tensor2Gen, "argmin")("jnp.argmin(t)", _.argmin)
  checkReductionOpsToFloat(tensor3Gen, "argmin")("jnp.argmin(t)", _.argmin)

  checkReductionOpsToBool(toBoolTensor(twoTensor0Gen), "all")("jnp.all(t)", _.all)
  checkReductionOpsToBool(toBoolTensor(twoTensor1Gen), "all")("jnp.all(t)", _.all)
  checkReductionOpsToBool(toBoolTensor(twoTensor2Gen), "all")("jnp.all(t)", _.all)
  checkReductionOpsToBool(toBoolTensor(twoTensor3Gen), "all")("jnp.all(t)", _.all)

  checkReductionOpsToBool(toBoolTensor(twoTensor0Gen), "any")("jnp.any(t)", _.any)
  checkReductionOpsToBool(toBoolTensor(twoTensor1Gen), "any")("jnp.any(t)", _.any)
  checkReductionOpsToBool(toBoolTensor(twoTensor2Gen), "any")("jnp.any(t)", _.any)
  checkReductionOpsToBool(toBoolTensor(twoTensor3Gen), "any")("jnp.any(t)", _.any)

  // Approx equal test
  property("approxEquals Tensor[a, b]"):
    forAll(tensor2Gen): t1 =>
      val t2 = t1 :* Tensor0(1 + Float.MinValue)
      val pyRes = {
        py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
        py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
        py.exec(s"res = jnp.allclose(t1, t2)")
        py.eval("res.item()").as[Boolean]
      }
      val scalaRes = t1.approxEquals(t2)
      pyRes shouldEqual scalaRes

  private def pythonScalaBinaryReductionOpsToBool[T <: Tuple : Labels](t1: Tensor[T], t2: Tensor[T])(
    pythonProgram: String,
    scalaProgram: (Tensor[T], Tensor[T]) => Boolean,
  ): (Boolean, Boolean) =
    require(t1.shape == t2.shape, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")
    val pyRes = {
      py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
      py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Boolean]
    }
    val scalaRes = scalaProgram(t1, t2)
    (pyRes, scalaRes)

  private def pythonScalaReductionOpsToFloat[T <: Tuple : Labels](t: Tensor[T])(
    pythonProgram: String,
    scalaProgram: Tensor[T] => Tensor0,
  ): (Float, Float) =
    val pyRes = {
      py.eval("globals()").bracketUpdate("t", t.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Float]
    }
    val scalaRes = scalaProgram(t).toFloat
    (pyRes, scalaRes)

  private def pythonScalaReductionOpsToBool[T <: Tuple : Labels](t: Tensor[T])(
    pythonProgram: String,
    scalaProgram: Tensor[T] => Boolean,
  ): (Boolean, Boolean) =
    val pyRes = {
      py.eval("globals()").bracketUpdate("t", t.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Boolean]
    }
    val scalaRes = scalaProgram(t)
    (pyRes, scalaRes)
