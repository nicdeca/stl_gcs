## Algorithm steps

- Define the STL formulas, made of operators, corresponding times for temporal operators and predicate functions. Defined as element of the Formula class

- Given the formula associate to each operator a polytope according to the operator2polytope rules and build the graph

- Give as input to BsplineGraphOfConvexSets the polytopes representing the the time-varying sets satisfying the task along with the edges connecting the polytopes. Vertices coordinates are the Control points of the Bspline in space and time