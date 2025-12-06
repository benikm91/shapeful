package src.main.scala.basicv2

import shapeful.Label
import shapeful.tensorv2.{Axis, AxisIndex, Shape, Tensor1, Tensor2, Tensor, DType, Device, NameOf}
import shapeful.tensorv2.TensorOps.*
import shapeful.tensorv2.TupleHelpers.{Remover, RemoverAll}
import shapeful.tensorv2.Axis.UnwrapAxes


@main
def linearLayerExamples(): Unit = 
    specificLinearLayerExample()
    generalLinearLayerExample()

/**
 * A simple linear layer example with fixed axis labels and 2D input.
 */
def specificLinearLayerExample(): Unit =
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

/**
 * A more general linear layer, with general axes and capability to handle arbitrary dimensional inputs.
 */
def generalLinearLayerExample(): Unit =
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
        ): Tensor[Tuple.Concat[remover.Out, Tuple1[OutputAxis]]] =
            forward[T, ContractAxis](Axis[ContractAxis])(input)

        def forward[T <: Tuple : NameOf, NewContractAxis <: Label : ValueOf](axis: Axis[NewContractAxis])(
            input: Tensor[T],
        )(
            using 
            axisIndex: AxisIndex[T, NewContractAxis],
            remover: Remover[T, NewContractAxis],
        ): Tensor[Tuple.Concat[remover.Out, Tuple1[OutputAxis]]] =
            import NameOf.ForConcat.given

            val newWeight: Tensor2[NewContractAxis, OutputAxis] = weight.as[(Axis[NewContractAxis], Axis[OutputAxis])]
            val out = input.contract(Axis[NewContractAxis])(newWeight)

            out.vapply(Axis[OutputAxis]){ _ + bias }
    
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