package src.main.scala.basic

import shapeful.StringMath.*
import shapeful.tensorv2.{Axis, Shape, Tensor1, Tensor2, Tensor, DType, Device}
import scala.collection.compat.immutable.ArraySeq
import shapeful.tensorv2.TensorOps.*
import shapeful.tensorv2.StatisticOps.*
import shapeful.tensorv2.TupleHelpers
import shapeful.tensorv2.TupleHelpers.NameOf
import shapeful.tensorv2.TupleHelpers.UnwrapAxes
import shapeful.tensorv2.TupleHelpers.ValuesOf
import shapeful.tensorv2.AxisIndex
import shapeful.tensorv2.TupleHelpers.ValuesOf.AxesFactory
import shapeful.tensorv2.Remover
import scala.collection.View.Zip

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
    import shapeful.tensorv2.Tensor.{Tensor1, Tensor2}
    case class LinearLayer(
      weight: Tensor2["Feature", "Output"],
      bias: Tensor1["Output"],
    ):
      def forward(
        input: Tensor2["Batch", "Feature"],
      ): Tensor2["Batch", "Output"] =
        val out = input.contract(Axis["Feature"])(weight)
        out.vmap(Axis["Batch"]){ _ + bias }
    val layer = LinearLayer(
      weight = Tensor.zeros(Shape(
        Axis["Feature"] -> 16,
        Axis["Output"] -> 2,
      )),
      bias = Tensor.zeros(Shape(
        Axis["Output"] -> 2,
      )),
    )
    val input = Tensor.zeros(Shape(
      Axis["Batch"] -> 32,
      Axis["Feature"] -> 16,
    ))
    val output = layer.forward(input)
    println(output.shape)
  }
  {
    println("Contraction with overlapping axes")
    import shapeful.Label
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
    import shapeful.Label
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
    import shapeful.Label
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
    import shapeful.Label
    import shapeful.tensorv2.Tensor.{Tensor1, Tensor2}

    case class LinearLayer[
      ContractAxis <: Label : ValueOf,
      OutputAxis <: Label : ValueOf,
    ](
      weight: Tensor2[ContractAxis, OutputAxis],
      bias: Tensor1[OutputAxis],
    ):
      def forward[T <: Tuple : NameOf](
        input: Tensor[T],
      )(
        using 
        axisIndex: AxisIndex[T, ContractAxis],
        remover: Remover[T, ContractAxis],
        axesFactory: AxesFactory[remover.Out],
      ): Tensor[Tuple.Concat[remover.Out, Tuple1[OutputAxis]]] =
        forward[T, ContractAxis](Axis[ContractAxis])(input)

      def forward[T <: Tuple : NameOf, NewContractAxis <: Label : ValueOf](axis: Axis[NewContractAxis])(
        input: Tensor[T],
      )(
        using 
        axisIndex: AxisIndex[T, NewContractAxis],
        remover: Remover[T, NewContractAxis],
        otherRemover: Remover[(NewContractAxis, OutputAxis), NewContractAxis],
        axesFactory: AxesFactory[remover.Out],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import NameOf.ForConcat.given

        val newWeight = weight.as[(Axis[NewContractAxis], Axis[OutputAxis])]
        val out = input.contract(Axis[NewContractAxis])(newWeight)
        
        val axes = axesFactory()
        out
        // TODO implement zip
    
    val layer = LinearLayer(
      weight = Tensor.zeros(Shape(
        Axis["Feature"] -> 16,
        Axis["Output"] -> 2,
      )),
      bias = Tensor.zeros(Shape(
        Axis["Output"] -> 2,
      )),
    )
    val output = layer.forward(Tensor.zeros(Shape(
      Axis["Batch"] -> 32,
      Axis["Feature"] -> 16,
    )))
    println(output.shape)
    val output2 = layer.forward(Tensor.zeros(Shape(
      Axis["Batch"] -> 32,
      Axis["Patient"] -> 4,
      Axis["Frame"] -> 8,
      Axis["Feature"] -> 16,
    )))
    println(output2.shape)
    val output3 = layer.forward(Tensor.zeros(Shape(
      Axis["Batch"] -> 32,
      Axis["Feature"] -> 16,
      Axis["Patient"] -> 4,
      Axis["Frame"] -> 8,
    )))
    println(output3.shape)
    val output4 = layer.forward(Axis["Value"])(Tensor.zeros(Shape(
      Axis["Batch"] -> 32,
      Axis["Patient"] -> 4,
      Axis["Frame"] -> 8,
      Axis["Value"] -> 16,
    )))
    println(output4.shape)
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
      Axis["C"] -> 4,
      Axis["D"] -> 5,
    )).slice((
      Axis["B"] -> 2,
      Axis["C"] -> 3,
    ))
    println(res2.shape)
  }
  {
    import shapeful.Label
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

    case class ZipResult[L <: Label : ValueOf, T1 <: Tuple : NameOf, T2 <: Tuple : NameOf](
      t1: Tensor[T1],
      t2: Tensor[T2],
    ):
      def vmap[OutShape <: Tuple : NameOf](
        using
        remover1: Remover[T1, L],
        remover2: Remover[T2, L],
      )(f: (Tensor[remover1.Out], Tensor[remover2.Out]) => Tensor[OutShape])(
        axis: Axis[L],
      )(
        using 
        axisIndex1: AxisIndex[T1, L],
        axisIndex2: AxisIndex[T2, L],
      ): Tensor[L *: OutShape] = 
        val dimSize = t1.shape.dimensions(axisIndex1.value)
        val res = (0 until dimSize).toList.map { i =>
          val slice1 = t1.slice(axis -> i)
          println(slice1.shape)
          val slice2 = t2.slice(axis -> i)
          f(slice1, slice2)
        }
        Tensor.stack(res, axis)

    def zip[L <: Label : ValueOf, T1 <: Tuple : NameOf, T2 <: Tuple : NameOf](zipAxis: Axis[L])(
      t1: Tensor[T1],
      t2: Tensor[T2],
    ): ZipResult[L, T1, T2] = ZipResult(t1, t2)

    val res = zip(Axis[Batch])(x, y).vmap((xi, yi) => xi.sum + yi.sum)(Axis[Batch])
    println(res.shape)
  }