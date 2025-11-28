package src.main.scala.basic

import shapeful.tensorv2.{Axis, Shape, Tensor1, Tensor2, Tensor, DType, Device}
import scala.collection.compat.immutable.ArraySeq
import shapeful.tensorv2.TensorOps.*
import shapeful.tensorv2.TupleHelpers
import shapeful.tensorv2.TupleHelpers.ShapeTreeOf
import shapeful.tensorv2.TupleHelpers.TupleFlat
import shapeful.tensorv2.TupleHelpers.UnwrapAxes

def main(args: Array[String]): Unit =
  println("TensorV2 Playground")
  {
    println("MatMul tests")
    val values = ArraySeq(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    ).map(_.toFloat)
    val X = Tensor2(
      values = values,
      shape = Shape(
        Axis["Samples"] -> 10,
        Axis["Features"] -> 2,
      )
    )
    val XT = X.transpose
    val XTX = XT.matmul(X)
    val XXT = X.matmul(XT)
    println(XTX.shape)
    println(XXT.shape)
  }
  {
    println("Normalization example")
    val values = ArraySeq(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    ).map(_.toFloat)
    val X = Tensor2(
      values = values,
      shape = Shape(
        Axis["Samples"] -> 10,
        Axis["Features"] -> 2,
      )
    )
    val means = X.vmap(Axis["Features"])(_.mean)
    val stds = X.vmap(Axis["Features"])(_.std)
    val Xnorm = X.vmap(Axis["Samples"]){ (x) =>
      (x - means) / stds
    }
    println(Xnorm)
    println(Xnorm.shape)
    println(Xnorm.device)
    println(Xnorm.dtype)
  }
  {
    println("DType and Device tests")
    val t = Tensor.zeros(Shape(
      Axis["Batch"] -> 1024,
      Axis["Features"] -> 512,
    ))
    println(t.shape)
    println(t.dtype)
    println(t.asType(DType.Int32).dtype)
    println(t.device)
    println(t.toDevice(Device.CPU).device)
  }
  {
    val x = Tensor.zeros(Shape(
      Axis["Features"] -> 2,
    ))
    val A = Tensor.zeros(Shape(
      Axis["Samples"] -> 50,
      Axis["Features"] -> 2,
    ))
    // val y1 = x.contract(Axis["A"])(A)
    val y1 = A.contract(Axis["Features"])(x)
    println(y1.shape)
    // A.contract(Axis["lala"])(x)
    // A.contract(Axis["Samples"])(x)
    val y2 = x.contract(Axis["Features"])(A)
    println(y2.shape)
    val y3 = x.outerProduct(A)
    println(y3.shape)
  }
  {
    println("Einops rearrange tests")
    type Batch = "batch"
    type Frame = "frame"
    type BatchFrame = "batch_frame"
    type Width = "width"
    type Height = "height"
    type Channel = "channel"
    val X = Tensor.zeros(Shape(
      Axis[Batch] -> 32,
      Axis[Frame] -> 64,
      Axis[Width] -> 256,
      Axis[Height] -> 256,
      Axis[Channel] -> 3,
    ))
    val d = X.rearrange(
      (
        Axis[(Batch, Frame)],
        Axis[(Width, Height)],
        Axis[Channel]
      )
    )
    println(d.shape)
    val e = d.rearrange(
      (Axis[Batch], Axis[Frame], Axis[Width], Axis[Height], Axis[Channel]),
      (
        Axis[Batch] -> 32,
        Axis[Frame] -> 64,
        Axis[Width] -> 256,
        Axis[Height] -> 256,
      )
    )
    println(e.shape)
    val f = d.rearrange(
      (Axis[Batch], Axis[Frame], (Axis[(Width, Height)]), Axis[Channel]),
      (
        Axis[Batch] -> 32,
        Axis[Frame] -> 64,
      )
    )
    println(f.shape)
    val g = d.relabel(Axis[(Batch, Frame)], Axis[Frame])
    println(g.shape)
  }