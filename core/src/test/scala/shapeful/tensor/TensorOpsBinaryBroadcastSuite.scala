package shapeful.tensor

import shapeful.*
import shapeful.random.Random
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

class TensorOpsBinaryBroadcastSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:
  
  py.exec("import jax.numpy as jnp")
  val testKey = Random.Key(42)
  
  def abcdGen: Gen[(Int, Int, Int, Int)] = for {
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    } yield (a, b, c, d)

  def aAbcdGen: Gen[(Tensor1[A], Tensor[(A, B, C, D)])]= for {
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    } yield (
      Tensor.randn(Shape(Axis[A] -> a), testKey),
      Tensor.randn(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), testKey),
    )

  def abAbcdGen: Gen[(Tensor2[A, B], Tensor[(A, B, C, D)])]= for {
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    } yield (
      Tensor.randn(Shape(Axis[A] -> a, Axis[B] -> b), testKey),
      Tensor.randn(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), testKey),
    )

  def dAbcdGen: Gen[(Tensor1[D], Tensor[(A, B, C, D)])] = for {
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    } yield (
      Tensor.randn(Shape(Axis[D] -> d), testKey),
      Tensor.randn(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), testKey),
    )

  def cdAbcdGen: Gen[(Tensor2[C, D], Tensor[(A, B, C, D)])] = for {
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    } yield (
      Tensor.randn(Shape(Axis[C] -> c, Axis[D] -> d), testKey),
      Tensor.randn(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), testKey),
    )

  def bcdAbcdGen: Gen[(Tensor[(B, C, D)], Tensor[(A, B, C, D)])] = for {
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    } yield (
      Tensor.randn(Shape(Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), testKey),
      Tensor.randn(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), testKey),
    )

  property("Broadcasting matches vapply: a + abcd"):
    forAll(aAbcdGen): (a, abcd) => 
      val broadcastResult = a +: abcd
      val vapplyResult = abcd.vapply(Axis[A])(ai => ai + a)
      broadcastResult should approxEqual(vapplyResult)

  property("Broadcasting matches vapply: d + abcd"):
    forAll(dAbcdGen): (d, abcd) => 
      val broadcastResult = d +: abcd
      val vapplyResult = abcd.vapply(Axis[D])(di => di + d)
      broadcastResult should approxEqual(vapplyResult)

  property("Broadcasting matches vmap: bcd + abcd"):
    forAll(bcdAbcdGen): (bcd, abcd) => 
      val broadcastResult = bcd +: abcd
      val cvmapResult = abcd.vmap(Axis[A])(bcdi => bcdi + bcd)
      broadcastResult should approxEqual(cvmapResult)

  property("Broadcasting matches vmap: cd + abcd"):
    forAll(cdAbcdGen): (cd, abcd) => 
      val broadcastResult = cd +: abcd
      val cvmapResult = abcd.vmap(Axis[A])(_.vmap(Axis[B])(cdi => cdi + cd))
      broadcastResult should approxEqual(cvmapResult)