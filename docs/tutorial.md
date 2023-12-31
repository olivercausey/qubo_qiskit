# Traveling Salesman Problem (TSP) Tutorial using Qiskit

In this tutorial, we'll explore how to model and solve a simplified version of the Traveling Salesman Problem (TSP) as a Quadratic Unconstrained Binary Optimization (QUBO) problem using Qiskit. TSP is a classic logistics problem where a salesman needs to visit each city exactly once and return to the starting city while minimizing the total travel distance.

## Formulating the Traveling Salesman Problem as a QUBO

### Problem Definition:
- We have \( n \) cities and the distance between each pair of cities is known.
- The goal is to find the shortest possible route that visits each city exactly once and returns to the origin city.

### QUBO Formulation:
- We'll represent the problem using binary variables \( x_{i,j} \) where \( x_{i,j} = 1 \) if the salesman travels from city \( i \) to city \( j \), and \( x_{i,j} = 0 \) otherwise.
- The objective is to minimize the total distance traveled, which can be written as:
\[ \text{Minimize } \sum_{i,j} d_{i,j} \cdot x_{i,j} \]
where \( d_{i,j} \) is the distance between city \( i \) and city \( j \).

### Constraints:
- Each city should be visited exactly once:
\[ \sum_{i} x_{i,j} = 1, \quad \sum_{j} x_{i,j} = 1, \quad \forall i, j \]
- Subtour elimination (to ensure the salesman doesn’t visit a subset of cities and return to the starting point, ignoring other cities): A common way is to introduce auxiliary variables and additional constraints to the QUBO formulation.

### Combining Objective and Constraints:
- We can combine the objective function and constraints into a single QUBO expression by introducing penalty terms for violating the constraints:
\[ \text{Minimize } \sum_{i,j} d_{i,j} \cdot x_{i,j} + P \cdot \left( \sum_{i} x_{i,j} - 1 \right)^2 + P \cdot \left( \sum_{j} x_{i,j} - 1 \right)^2 + \ldots \]
where \( P \) is a sufficiently large penalty term.

### Encoding the QUBO into a Matrix:
- Finally, we need to encode the QUBO formulation into a matrix format which can be input into Qiskit for solving. This involves creating a matrix \( Q \) where each element \( Q_{ij} \) corresponds to the quadratic coefficients of the binary variables \( x_{i,j} \).
