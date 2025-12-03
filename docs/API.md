# ScalaJAX - API

## Overview

We separate operations logically into modules declared in `TensorOps.scala`, note that `Tensor.scala` defines no operations. We will discuss each module down below.
From the user's perspective these modules are hidden, all operations are available using:
```scala mdoc
import shapeful.tensorv2.TensorOps.*
```

## Elementwise operations

Operations working on an element-by-element basis, i.e., the shape of the tensor does not change.

TODO

## Reduction operations

Operations summarizing values in the tensor, i.e., the shape of the tensor does reduced.

TODO

## Contraction operations

Operations working on the shape of tensors, i.e., the shape of the tensor does change.

TODO

## Structural operations

TODO

## Functional operations

TODO

## (Common) LinearAlgebra operations

TODO

## (Common) Statistical operations

TODO

## Special cases

TODO

### Scalar operations

TODO

### Vector operations

TODO

### Matrix operations

TODO
