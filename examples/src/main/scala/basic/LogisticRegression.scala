package examples.basic

import shapeful.*
import shapeful.autodiff.*
import shapeful.nn.*
import shapeful.jax.Jax
import shapeful.random.Random
import shapeful.optimization.GradientDescent
import examples.DataUtils
import io.github.quafadas.table.*
import scala.compiletime.constValueTuple
import scala.collection.immutable.ArraySeq

object LogisticRegression:

  type Sample = "sample"
  type Feature = "feature"
  type Output = "output"

  case class Params(
      weights: Tensor1[Feature],
      bias: Tensor0
  ) derives TensorTree,
        ToPyTree

  def initParams(key: Random.Key)(using featureDim: Dim[Feature]): Params =
    val keys = key.split(2)
    Params(
      weights = Tensor.zeros(Shape(Axis[Feature] -> featureDim.dim)),
      bias = Tensor.zeros(Shape0)
    )

  def forward(params: Params, x: Tensor1[Feature]): (Tensor0, Tensor0) =
    val logits = x.dot(params.weights) + params.bias
    val probs = logits.sigmoid
    (logits, probs)

  def main(args: Array[String]): Unit =

    import io.github.quafadas.table.*
    val df = CSV
      .resource("penguins.csv", TypeInferrer.FromAllRows)
      .filter(row => !(row.species == 2))
      .toSeq

    val dfShuffled = scala.util.Random.shuffle(df)

    val learningRate = 3e-1f
    val key = Random.Key(42)

    val numFeatures = 4
    given Dim[Feature] = Dim(numFeatures)

    val (dataKey, trainKey) = Random.Key(42).split2()
    val featureData = dfShuffled
      .map { row =>
        Array(
          row.flipper_length_mm.toFloat,
          row.bill_length_mm.toFloat,
          row.bill_depth_mm.toFloat,
          row.body_mass_g.toFloat
        )
      }
      .toArray
      .flatten
    val labelData = dfShuffled.column["species"].toArray.map(_.toFloat)

    val dataUnnormalized = Tensor2.fromArray(
      Shape(Axis[Sample] -> df.length, Axis[Feature] -> numFeatures),
      ArraySeq.unsafeWrapArray(featureData)
    )
    val numTrainSamples = (df.length * 80) / 100
    val (trainingDataUnnormalized, valDataUnnormalized) = dataUnnormalized.split(Axis[Sample], numTrainSamples)
    val dataLabels = Tensor1.fromArray(Axis[Sample], ArraySeq.unsafeWrapArray(labelData))
    val (trainLabels, valLabels) = dataLabels.split(Axis[Sample], numTrainSamples)

    def calcMeanAndStd(t: Tensor2[Sample, Feature]): (Tensor1[Feature], Tensor1[Feature]) =
      val mean = t.vmap(Axis[Feature]) { _.mean }
      val std = t.zipVmap(Axis[Feature])(mean) { (x, m) => (x - m).pow(Tensor0(2f)).mean.sqrt + Tensor0(1e-6f) }
      (mean, std)

    def standardize(mean: Tensor1[Feature], std: Tensor1[Feature])(t: Tensor2[Sample, Feature]): Tensor2[Sample, Feature] =
      t.vmap(Axis[Sample]) { (x) => (x - mean) / std }

    val (trainMean, trainStd) = calcMeanAndStd(trainingDataUnnormalized)
    val trainingData = standardize(trainMean, trainStd)(trainingDataUnnormalized)
    val valData = standardize(trainMean, trainStd)(valDataUnnormalized)

    val (initKey, restKey) = trainKey.split2()
    val (lossKey, sampleKey) = restKey.split2()

    def loss(data: Tensor2[Sample, Feature])(p: Params): Tensor0 =
      val losses = data.zipVmap(Axis[Sample])(trainLabels) { (sample, label) =>
        val (logits, probs) = forward(p, sample)
        (logits.relu - logits * label + ((logits.abs * Tensor0(-1f)).exp + Tensor0(1f)).log)
      }
      losses.mean

    val initialParams = initParams(initKey)

    val trainLoss = loss(trainingData)
    val valLoss = loss(valData)
    val gradFn = Autodiff.grad(trainLoss)
    val gd = GradientDescent(learningRate)
    val finalParams = (1 to 2500)
      .foldLeft(initialParams) { (params, i) =>
        if i % 10 == 0 then
          val trainOutputs = trainingData.vmap(Axis[Sample]) { x => forward(params, x)._2 }
          val valOutputs = valData.vmap(Axis[Sample]) { x => forward(params, x)._2 }
          println(List(
            "trainAcc: " + (Tensor0(1f) - (trainOutputs - trainLabels).abs.mean),
            "valAcc: " + (Tensor0(1f) - (valOutputs - valLabels).abs.mean)
          ).mkString(", "))
        end if
        gd.step(gradFn, params)
      }

    val predictions = trainingData.vmap(Axis[Sample]) { x => forward(finalParams, x)._2 }
    println(predictions)
    val predictionClasses = predictions.vmap(Axis[Sample]) { p => p.argmax }

    println("\nTraining complete. Optimized parameters:" + finalParams)
