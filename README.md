# mapper-tda

A reasonably decent implementation of the mapper algorithm from TDA

## What is TDA?

TDA stands for "Topological Data Analysis", a branch of data analysis using topological tools to recover insights from datasets. 

## The Mapper Algorithm 

Assume we have a dataset D inside a metric space X, together with the following choices:
1. A continuous map f:X -> Y 
2. A cover algorithm for f(D)
3. A clustering algorithm for D.

The mapper algorithm follows these steps:
1. Build an open cover of f(D)
2. For each open chart U of f(D) let V the preimage of U under f, then the V's form an open cover of D. For each V, run the chosen clustering algorithm
3. For each local cluster obtained, build a node. Whenever two local clusters (from different V's) intersect, draw an edge between their corresponding nodes.

The graph obtained is called a "mapper graph".