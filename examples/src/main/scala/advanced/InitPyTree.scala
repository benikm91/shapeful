package examples.advanced

import shapeful.*
import shapeful.autodiff.*

object InitPyTree extends App:

  println("=== Custom PyTree Examples ===\n")

  type Feature = "feature"
  type Hidden = "hidden"
  type Output = "output"
  type Layer = "layer"

  // 1. Simple custom structure
  println("1. Simple Custom PyTree Structure")

  case class SimpleModel(
      weight: Tensor2[Feature, Hidden],
      bias: Tensor1[Hidden],
      scale: Tensor0
  ) derives TensorTree, ToPyTree
