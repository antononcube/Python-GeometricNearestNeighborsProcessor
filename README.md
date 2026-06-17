# GeometricNearestNeighborsProcessor

A Python package for Geometric Nearest Neighbors (GNN) workflows: data rescaling, fast anomalies finding, similarity matrices derivation.

This Python package is a translation to Python of the Wolfram Language software monad 
["MonadicGeometricNearestNeighbors"](https://resources.wolframcloud.com/PacletRepository/resources/AntonAntonov/MonadicGeometricNearestNeighbors), [AAp1].
(The R package ["GNNMon-R"](https://github.com/antononcube/R-packages/tree/master/GNNMon-R), [AAp4], is another translation of [AAp1].)
 

----

## Usage examples

Load packages:

```python
from RandomDataGenerators import *
from OutlierIdentifiers import *

import numpy
import random

import pandas as pd
import plotly.express as px
```


Generate random points:

```python
help(random_data_frame)
```

```python
dfPoints = random_data_frame(n_rows=30, columns_spec = ["X", "Y"], generators= {"X": numpy.random.normal, "Y": numpy.random.normal})
print(dfPoints.shape)
print(dfPoints[1:6])
```

Here is a summary:

```python
dfPoints.describe()
```

Here is a plot of the points:

```python
fig = px.scatter(dfPoints, x="X", y="Y", template="plotly_dark")
fig.show()
```

A typical pipeline of geometric nearest neighbors processing:

```python
gnnObj = (GeometricNearestNeighborsProcessor(dsPoints)
   .make_nearest_function(distance_function = "EuclideanDistance")
   .compute_thresholds(number_of_nearest_neighbors = 10, aggregation_function = "mean", outlier_identifier = "QuartileIdentifierParameters")
   .find_anomalies()
   .echo_function_value("Anomaly points:", lambda x: print(x))
   .plot(title="Random points", template="plotly_dark")
)
```

Show the plot obtained above:

```python
gnnObj.take_value().show()
```

Here we generate another set of random points using the same random point generators:

```python
dfPoints2 = random_data_frame(n_rows=40, columns_spec = ["X", "Y"], generators= {"X": numpy.random.normal, "Y": numpy.random.normal})
print(dfPoints2.shape)
```

Here the points of second set are classified into being anomalous or not:

```python
gnnObj.classify(dfPoints2).take_value()
```

----

## References

### Wolfram Language

[AAp1] Anton Antonov, [MonadicGeometricNearestNeighbors](https://resources.wolframcloud.com/PacletRepository/resources/AntonAntonov/MonadicGeometricNearestNeighbors), Wolfram Language paclet, 
(2023-2025), 
[Wolfram Language Paclet Repository](https://resources.wolframcloud.com/PacletRepository).

[AAp2] Anton Antonov, [OutlierIdentifiers](https://resources.wolframcloud.com/PacletRepository/resources/AntonAntonov/OutlierIdentifiers/), Wolfram Language paclet, 
(2023), 
[Wolfram Language Paclet Repository](https://resources.wolframcloud.com/PacletRepository).

### R

[AAp3] Anton Antonov, [OutlierIdentifiers](https://github.com/antononcube/R-packages/tree/master/OutlierIdentifiers), R package,
(2019-2024),
[GitHub/antononcube](https://github.com/antononcube).

[AAp4] Anton Antonov, [GNNMon-R](https://github.com/antononcube/R-packages/tree/master/GNNMon-R), R package,
(2019-2025),
[GitHub/antononcube](https://github.com/antononcube).

[AAp5] Anton Antonov, [KDTreeAlgorithm](https://github.com/antononcube/R-packages/tree/master/KDTreeAlgorithm), R package,
(2025),
[GitHub/antononcube](https://github.com/antononcube).

### Python

[AAp6] Anton Antonov, [RandomDataGenerators](https://github.com/antononcube/Python-packages/tree/main/RandomDataGenerators), Python package,
(2021-2026), 
[GitHub/antononcube](https://github.com/antononcube).
([PIPy.org](https://pypi.org/project/RandomDataGenerators/).)

[AAp7] Anton Antonov, [OutlierIdentifiers](https://github.com/antononcube/Python-packages/tree/main/OutlierIdentifiers), Python package,
(2024), 
[GitHub/antononcube](https://github.com/antononcube).
([PIPy.org](https://pypi.org/project/OutlierIdentifiers/).)