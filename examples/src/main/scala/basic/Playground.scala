package src.main.scala.basic

import shapeful.tensorv2.{Axis, Shape, Tensor2}
import scala.collection.compat.immutable.ArraySeq
import shapeful.tensorv2.TensorOps.*

def main(args: Array[String]): Unit =
  println("TensorV2 Playground")
  {
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
    val XT = X.rearrange(
      Axis["Features"],
      Axis["Samples"],
    )
    val XTX = XT.matmul(X)
    val XXT = X.matmul(XT)
    }
  {
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
