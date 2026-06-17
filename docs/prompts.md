# Prompts 

## Introduction

This file has Large Language Model (LLM) prompts that can be helpful for the workflows of "GeometricNearestNeighborsProcessor" 
while working in Jupyter notebooks.

The magic cell specs below are included to simplify the prompts usage with the package ["JupyterChatbook"](https://pypi.org/project/JupyterChatbook/).

---

## Programming

The initial version of the package was made by LLM-reprogramming of the 
[code](https://github.com/antononcube/R-packages/blob/master/GNNMon-R/R/GNNMonFunctions.R) 
of the R package
[GNNMon-R](https://github.com/antononcube/R-packages/tree/master/GNNMon-R).

Note that the [README](../README.md) of this (Python) package was already written, i.e., it was used as a spec.

```
Analyze the R package in "./R/GNNMonFunctions.R" and make the corresponding Python package in this directory. 
The Python package should provide the class GeometricNearestNeighborsProcessor. 
See the "./README.md" file for the basic usage. Use snake case for the methods and their arguments. 
For example, the R function GNNMonComputeThresholds should become the Python class method compute_thresholds. 
Use the Python package "OutlerIdentifiers" for the outlier identifier functions. 
Use a popular Python package for the finding the nearest neighbors of geometric points.
```

---

## Plot points

```
%%chat -i=python
I have two data frames with columns "X" and "Y". All rows of the second are also in the first.
I want to have a plotly scatter plot in which the second points are highlighted.
```

```
%%chat -i=python
Can you make the second points to be a smaller radius and to be seen that they lie on top of the first ones.
Use the dark mode template and "X" and "Y" as axes labels. 
```