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
      
  def checkReductionOps[T <: Tuple : Labels](gen: Gen[(Tensor[T], Tensor[T])], suffix: String)(pyCode: String, scOp: (Tensor[T], Tensor[T]) => Boolean) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) => 
        val (py, sc) = pythonScalaReductionOps(t1, t2)(pyCode, scOp)
        py shouldEqual sc

  checkReductionOps(twoTensor0Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkReductionOps(twoSameTensor0Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkReductionOps(twoTensor1Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  // checkReductionOps(twoSameTensor1Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkReductionOps(twoTensor2Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  // checkReductionOps(twoSameTensor2Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkReductionOps(twoTensor3Gen, "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  // checkReductionOps(twoSameTensor3Gen, "== (same)")("jnp.array_equal(t1, t2)", _ == _)

  private def pythonScalaReductionOps[T <: Tuple : Labels](t1: Tensor[T], t2: Tensor[T])(
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
