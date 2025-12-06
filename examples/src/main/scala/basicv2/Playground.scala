package src.main.scala.basicv2

import scala.collection.compat.immutable.ArraySeq
import shapeful.StringMath.*
import shapeful.Label
import shapeful.tensorv2.{Axis, AxisIndex, Shape, Tensor1, Tensor2, Tensor, DType, Device, NameOf}
import shapeful.tensorv2.TensorOps.*
import shapeful.tensorv2.TupleHelpers.{Remover, RemoverAll}
import shapeful.tensorv2.Axis.UnwrapAxes


@main def playground(): Unit =
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
        Axis[Batch * Frame],
        Axis[Width * Height],
        Axis[Channel]
      )
    )
    println(d.shape)
    val e = d.as[(Axis[Frame], Axis["pixel"], Axis[Channel])]
    println(e.shape)
  }
  {
    println("Contraction with overlapping axes")
    import scala.util.NotGiven
    def f[L1 <: Label : ValueOf, L2 <: Label : ValueOf, L3 <: Label : ValueOf](
      x: Tensor[(L1, L2)], 
      y: Tensor[(L2, L3)]
    ): Tensor[(L1, L3, L2)] = 
      x.vmap(Axis[L1]){ xi => 
        y.vmap(Axis[L3]){ yi =>
          xi + yi
        }
      }
    val z = f(Tensor.zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )), Tensor.zeros(Shape(
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )))
    println(z.shape)
  }
  {
    def f(t1: Tensor[("A", "C")], t2: Tensor[Tuple1["C"]]): Tensor[Tuple1["A"]] =
      t1.matmul(t2)
    val t1 = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["C"] -> 2,
    ))
    val t2 = Tensor.ones(Shape(
      Axis["C"] -> 2,
    ))
    println(f(t1, t2))
    println("vmap 2")
    import scala.util.NotGiven
    val x1 = Tensor.ones(Shape(
      Axis["B"] -> 1,
      Axis["A"] -> 2,
      Axis["C"] -> 2,
    ))
    val x2 = Tensor.ones(Shape(
      Axis["B"] -> 1,
      Axis["C"] -> 2,
    ))
  }
  {
    def f[L1 <: Label : ValueOf, L2 <: Label : ValueOf, L3 <: Label : ValueOf](x: Tensor[(L1, L2)], y: Tensor[(L2, L3)]) = 
      x.vmap(Axis[L1]){ xi =>
        y.vmap(Axis[L3]){ yi =>
          xi + yi
        }
      }
    println(f(
      Tensor.zeros(Shape(
        Axis["A"] -> 2,
        Axis["B"] -> 3,
      )),
      Tensor.zeros(Shape(
        Axis["B"] -> 3,
        Axis["C"] -> 4,
      ))
    ).shape)
  }

  {
    println("Ravel")
    val res = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).ravel
    println(res.shape)
  }
  {
    println("swapaxes")
    val res = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).swap(Axis["A"], Axis["C"])
    println(res.shape)
  }
  {
    println("appendAxis / prependAxis")
    val res = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).appendAxis(Axis["D"])
    println(res.shape)
    val res2 = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).prependAxis(Axis["D"])
    println(res2.shape)
  }
  {
    println("squeeze")
    val res = Tensor.ones(Shape(
      Axis["A"] -> 1,
      Axis["B"] -> 3,
      Axis["C"] -> 1,
    )).squeeze(Axis["A"])
    println(res.shape)
    val res2 = res.squeeze(Axis["C"])
    println(res2.shape)
  }
  {
    println("Slice")
    val res = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).slice(
      Axis["B"] -> 2
    )
    println(res.shape)
    val res2 = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).slice(
      Axis["B"] -> (0 to 1)
    )
    println(res2.shape)
    val res3 = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
      Axis["D"] -> 5,
    )).slice((
      Axis["B"] -> 2,
      Axis["C"] -> 3,
    ))
    println(res3.shape)
  }
  { 
    println("zipvmap tests")
    type Batch = "Batch"
    type Asset = "Asset"
    type Region = "Region"
    type Sector = "Sector"
    type Risk = "Risk"

    val x = Tensor.ones(Shape(
      Axis[Batch] -> 6,
      Axis[Asset] -> 3,
      Axis[Region] -> 5,
    ))

    val y = Tensor.ones(Shape(
      Axis[Region] -> 5,
      Axis[Batch] -> 6,
      Axis[Sector] -> 4,
    ))

    val z = Tensor.ones(Shape(
      Axis[Sector] -> 4,
      Axis[Risk] -> 5,
      Axis[Batch] -> 6,
    ))

    val res = zipvmap(Axis[Batch])(x, y) {
      case (xi, yi) => xi.sum + yi.sum
    }
    println(res.shape)

    val res2 = zipvmap(Axis[Batch])(x, y, z) {
      (xi, yi, zi) => xi.sum + yi.sum + zi.sum
    }
    println(res2.shape)
  }
  {
    import shapeful.tensorv2.* // Assuming imports

    type Batch = "Batch"
    type Asset = "Asset"
    type Region = "Region"
    type Sector = "Sector"
    type Risk = "Risk"

    val x = Tensor.ones(Shape(Axis[Batch] -> 6, Axis[Asset] -> 3, Axis[Region] -> 5))
    val y = Tensor.ones(Shape(Axis[Region] -> 5, Axis[Batch] -> 6, Axis[Sector] -> 4))
    val z = Tensor.ones(Shape(Axis[Sector] -> 4, Axis[Risk] -> 5, Axis[Batch] -> 6))

    val res = zipvmap(Axis[Batch])((x, y, z)) { 
      case (xi, yi, zi) => xi.sum + yi.sum + zi.sum 
    }
    println(res.shape)  
  }
  {
    println("TensorWhere tests")
    val x = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val y = Tensor.zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val condition = Tensor.zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).asType(DType.Bool)
    val res = where(condition, x, y)
    println(res.shape)
  }
  {
    println("Diag")
    val x = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val res = x.diagonal
    println(res.shape)
  }
  {
    import shapeful.tensorv2.Tensor0
    println("Set")
    val x = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).set((
      Axis["A"] -> 1,
      Axis["B"] -> 2,
    ))(Tensor0(42))
    println(x)
    val v = Tensor1(
      Axis["B"],
      ArraySeq(100, 101, 102),
    )
    val x2 = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).set(
      Axis["A"] -> 1,
    )(v)
    println(x2)
  }