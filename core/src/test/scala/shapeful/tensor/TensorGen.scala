package shapeful.tensor

import shapeful.*
import org.scalacheck.{Arbitrary, Gen}

object TensorGen:

  type A = "a"
  type B = "b"
  type C = "c"
  type D = "d"

  def genData(n: Int)(min: Float, max: Float): Gen[Array[Float]] = Gen.listOfN(n, Gen.choose(min, max)).map(_.toArray)

  def tensor0Gen: Gen[Tensor0] = tensor0GenOf(-1.0f, 1.0f)
  def tensor0GenOf(min: Float, max: Float): Gen[Tensor0] = 
    for {
      tensor0 <- tensor0GenOfShape(min, max)
    } yield tensor0

  def twoTensor0Gen: Gen[(Tensor0, Tensor0)] = twoTensor0GenOf(-1.0f, 1.0f)
  def twoTensor0GenOf(min: Float, max: Float): Gen[(Tensor0, Tensor0)] = 
    for {
      t1 <- tensor0GenOfShape(min, max)
      t2 <- tensor0GenOfShape(min, max)
    } yield (t1, t2)

  def twoSameTensor0Gen: Gen[(Tensor0, Tensor0)] = twoSameTensor0GenOf(-1.0f, 1.0f)
  def twoSameTensor0GenOf(min: Float, max: Float): Gen[(Tensor0, Tensor0)] = 
    for { 
      t1 <- tensor0GenOfShape(min, max) 
    } yield (t1, Tensor.fromPy(t1.jaxValue))

  def tensor0GenOfShape(min: Float, max: Float): Gen[Tensor0] = 
    for {
      value <- Gen.choose(min, max)
    } yield Tensor0(value)

  def tensor1Gen: Gen[Tensor1[A]] = tensor1GenOf(-1.0f, 1.0f)
  def tensor1GenOf(min: Float, max: Float): Gen[Tensor1[A]] = 
    for {
      len  <- Gen.choose(1, 100)
      tensor1 <- tensor1GenOfShape(len)(min, max)
    } yield tensor1

  def twoTensor1Gen: Gen[(Tensor1[A], Tensor1[A])] = twoTensor1GenOf(-1.0f, 1.0f)
  def twoTensor1GenOf(min: Float, max: Float): Gen[(Tensor1[A], Tensor1[A])] = 
    for {
      len  <- Gen.choose(1, 100)
      t1 <- tensor1GenOfShape(len)(min, max)
      t2 <- tensor1GenOfShape(len)(min, max)
    } yield (t1, t2)

  def twoSameTensor1Gen: Gen[(Tensor1[A], Tensor1[A])] = twoSameTensor1GenOf(-1.0f, 1.0f)
  def twoSameTensor1GenOf(min: Float, max: Float): Gen[(Tensor1[A], Tensor1[A])] = 
    for { 
      len  <- Gen.choose(1, 100)
      t1 <- tensor1GenOfShape(len)(min, max) 
    } yield (t1, Tensor.fromPy(t1.jaxValue))

  def tensor1GenOfShape(d1: Int)(min: Float, max: Float): Gen[Tensor1[A]] = 
    for {
      data <- genData(d1)(min, max)
    } yield Tensor1(Axis[A], data)
  
  def tensor2Gen: Gen[Tensor2[A, B]] = tensor2GenOf(-1.0f, 1.0f)
  def tensor2GenOf(min: Float, max: Float): Gen[Tensor2[A, B]] = 
    for {
      rows <- Gen.choose(1, 10)
      cols <- Gen.choose(1, 10)
      tensor2 <- tensor2GenOfShape(rows, cols)(min, max)
    } yield tensor2

  def tensor2SquareGen: Gen[Tensor2[A, B]] = tensor2SquareGenOf(-1.0f, 1.0f)
  def tensor2SquareGenOf(min: Float, max: Float): Gen[Tensor2[A, B]] = 
    for {
      dim <- Gen.choose(1, 10)
      tensor2 <- tensor2GenOfShape(dim, dim)(min, max)
    } yield tensor2

  def twoTensor2Gen: Gen[(Tensor2[A, B], Tensor2[A, B])] = twoTensor2GenOf(-1.0f, 1.0f)
  def twoTensor2GenOf(min: Float, max: Float): Gen[(Tensor2[A, B], Tensor2[A, B])] =
    for {
      rows <- Gen.choose(1, 10)
      cols <- Gen.choose(1, 10)
      t1 <- tensor2GenOfShape(rows, cols)(min, max)
      t2 <- tensor2GenOfShape(rows, cols)(min, max)
    } yield (t1, t2)

  def twoSameTensor2Gen: Gen[(Tensor2[A, B], Tensor2[A, B])] = twoSameTensor2GenOf(-1.0f, 1.0f)
  def twoSameTensor2GenOf(min: Float, max: Float): Gen[(Tensor2[A, B], Tensor2[A, B])] = 
    for { 
      rows <- Gen.choose(1, 10)
      cols <- Gen.choose(1, 10)
      t1 <- tensor2GenOfShape(rows, cols)(min, max)
    } yield (t1, Tensor.fromPy(t1.jaxValue))

  def tensor2GenOfShape(d1: Int, d2: Int)(min: Float, max: Float): Gen[Tensor2[A, B]] = 
    for {
      data <- genData(d1 * d2)(min, max)
    } yield Tensor2(Shape(Axis[A] -> d1, Axis[B] -> d2), data)

  def tensor3Gen: Gen[Tensor3[A, B, C]] = tensor3GenOf(-1.0f, 1.0f)
  def tensor3GenOf(min: Float, max: Float): Gen[Tensor3[A, B, C]] = 
    for {
      d1 <- Gen.choose(1, 5)
      d2 <- Gen.choose(1, 5)
      d3 <- Gen.choose(1, 5)
      tensor3 <- tensor3GenOfShape(d1, d2, d3)(min, max)
    } yield tensor3
  
  def twoTensor3Gen: Gen[(Tensor3[A, B, C], Tensor3[A, B, C])] = twoTensor3GenOf(-1.0f, 1.0f)
  def twoTensor3GenOf(min: Float, max: Float): Gen[(Tensor3[A, B, C], Tensor3[A, B, C])] =
    for {
      d1 <- Gen.choose(1, 5)
      d2 <- Gen.choose(1, 5)
      d3 <- Gen.choose(1, 5)
      t1 <- tensor3GenOfShape(d1, d2, d3)(min, max)
      t2 <- tensor3GenOfShape(d1, d2, d3)(min, max)
    } yield (t1, t2)

  def twoSameTensor3Gen: Gen[(Tensor3[A, B, C], Tensor3[A, B, C])] = twoSameTensor3GenOf(-1.0f, 1.0f)
  def twoSameTensor3GenOf(min: Float, max: Float): Gen[(Tensor3[A, B, C], Tensor3[A, B, C])] = 
    for { 
      d1 <- Gen.choose(1, 5)
      d2 <- Gen.choose(1, 5)
      d3 <- Gen.choose(1, 5)
      t1 <- tensor3GenOfShape(d1, d2, d3)(min, max)
    } yield (t1, Tensor.fromPy(t1.jaxValue))
  
  def tensor3GenOfShape(d1: Int, d2: Int, d3: Int)(min: Float, max: Float): Gen[Tensor3[A, B, C]] = 
    for {
      data <- genData(d1 * d2 * d3)(min, max)
    } yield Tensor3(Shape(Axis[A] -> d1, Axis[B] -> d2, Axis[C] -> d3), data)
