# Mirror_and_Gradient_Descent

This project explores the performance of Mirror Descent and Gradient Descent for solving a constrained optimization problem with an L1-regularized objective function. The goal is to compare their convergence behavior, regret, and robustness to noise, while providing insights into their theoretical foundations.

lgorithms:

Mirror Descent:
Uses KL-divergence as the Bregman divergence to project updates onto the simplex.
Naturally handles the simplex constraints, making it well-suited for this problem.
Achieves an optimal regret bound of 
O
(
T
)
O( 
T
​	
 ).
Adaptive Mirror Descent:
Dynamically adjusts the step size based on the gradient's magnitude.
Improves convergence by adapting to the problem's geometry.
Stochastic Mirror Descent:
Simulates real-world scenarios by adding noise to the gradients.
Maintains an 
O
(
T
)
O( 
T
​	
 ) regret bound even with noisy gradients.
Gradient Descent:
A classic baseline that uses Euclidean projections onto the simplex.
Serves as a comparison point to highlight the advantages of Mirror Descent

