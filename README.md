# mapper-tda
A reasonably decent implementation of the mapper algorithm from TDA

## What is TDA?
TDA stands for "Topological Data Analysis", a branch of data analysis using topological tools to recover insights from datasets. 

## The Mapper Algorithm 
Assume we have a dataset $D \subseteq X$ and the following choices:
1. A continuous map $f : X \to Y$ 
2. A cover algorithm for $f(D)$
3. A clustering algorithm for $D$.
The mapper algorithm follows these steps:
1. Build an open cover of $f(D)$
2. For each open chart $U \subseteq Y$ let $V = f^{-1}(U)$, then $D = \cup V$. For each $V$ run the chosen clustering algorithm
3. For each cluster build a node, and whenever two clusters intersect draw an edge between their corresponding nodes.

The graph obtained is a mapper graph