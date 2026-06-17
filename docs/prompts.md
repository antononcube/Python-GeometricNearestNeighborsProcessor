# Prompts 

## Introduction

This file has Large Language Model (LLM) prompts that can be helpful for the workflows of "GeometricNearestNeighborsProcessor" 
while working in Jupyter notebooks.

The magic cell specs below are included to simplify the prompts usage with the package ["JupyterChatbook"](https://pypi.org/project/JupyterChatbook/).

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