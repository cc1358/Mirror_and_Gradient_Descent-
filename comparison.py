import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the objective function f(x) = ||x - p||_2^2 + lambda * ||x||_1
def f(x, p=np.array([0.7, 0.3]), lambda_reg=0.1):
    """
    Objective function: Combines L2 distance to a target point p and L1 regularization.
    - L2 term: Encourages x to be close to p.
    - L1 term: Promotes sparsity in x.
    """
    return np.sum((x - p) ** 2) + lambda_reg * np.sum(np.abs(x))

# Gradient of f(x)
def grad_f(x, p=np.array([0.7, 0.3]), lambda_reg=0.1):
    """
    Gradient of the objective function.
    - L2 gradient: 2 * (x - p)
    - L1 subgradient: lambda_reg * sign(x) (Note: Non-differentiable at 0, but we use sign(x) for simplicity.)
    """
    return 2 * (x - p) + lambda_reg * np.sign(x)

# Bregman divergence (KL-divergence) for psi(x) = sum x_i log x_i
def bregman_div(x, y):
    """
    Bregman divergence with KL-divergence as the generating function.
    - Measures the "distance" between x and y in a non-Euclidean geometry.
    - Ensures updates respect the simplex constraints (sum(x) = 1, x >= 0).
    """
    x, y = np.array(x), np.array(y)
    return np.sum(x * np.log(x / y)) - np.sum(x) + np.sum(y)

# Mirror descent update step (projection onto simplex via KL-divergence)
def mirror_descent_step(x_t, grad, eta, max_iter=100, tol=1e-6):
    """
    Solves: x_{t+1} = argmin_{x in simplex} <eta * grad, x> + D_psi(x, x_t)
    - Uses KL-divergence to project onto the simplex.
    - Constraints: sum(x) = 1, x >= 0.
    """
    def objective(z):
        return eta * np.dot(grad, z) + bregman_div(z, x_t)
    
    # Constraints: simplex {x_1 + x_2 = 1, x >= 0}
    constraints = ({'type': 'eq', 'fun': lambda z: np.sum(z) - 1})
    bounds = [(0, None)] * len(x_t)
    
    # Initial guess: slightly perturb x_t to ensure feasibility
    x0 = np.clip(x_t, 1e-10, 1 - 1e-10)
    x0 = x0 / np.sum(x0)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': tol})
    return result.x

# Standard projected gradient descent for comparison
def project_simplex(y):
    """
    Projects y onto the simplex {x : sum(x) = 1, x >= 0}.
    - Used in gradient descent to enforce constraints.
    """
    u = np.sort(y)[::-1]
    rho = np.max(np.where(u + (1 - np.cumsum(u)) / (np.arange(len(y)) + 1) > 0)[0]) + 1
    theta = (1 - np.sum(u[:rho])) / rho
    return np.maximum(y + theta, 0)

def gradient_descent_step(x_t, grad, eta):
    """
    Gradient descent update step with projection onto the simplex.
    """
    z = x_t - eta * grad
    return project_simplex(z)

# Stochastic Mirror Descent
def stochastic_mirror_descent(x0, T, eta, noise_scale=0.1):
    """
    Stochastic Mirror Descent with noisy gradients.
    - noise_scale: Controls the magnitude of gradient noise.
    - Theoretical guarantee: Regret bound of O(sqrt(T)) even with noisy gradients.
    """
    x_t = np.array(x0)
    trajectory = [x_t.copy()]
    regrets = []
    f_star = f([0.7, 0.3])  # Approximate optimal value
    
    for t in range(T):
        grad = grad_f(x_t) + noise_scale * np.random.randn(*x_t.shape)  # Add noise to gradient
        eta_t = eta / np.sqrt(t + 1)  # Decreasing step size
        x_t = mirror_descent_step(x_t, grad, eta_t)
        trajectory.append(x_t.copy())
        regrets.append(f(x_t) - f_star)
    
    return np.array(trajectory), np.array(regrets)

# Mirror Descent Algorithm
def mirror_descent(x0, T, eta, adaptive=False):
    """
    Mirror Descent with optional adaptive step sizes.
    - adaptive: If True, uses adaptive step sizes for better convergence.
    - Theoretical guarantee: Regret bound of O(sqrt(T)).
    """
    x_t = np.array(x0)
    trajectory = [x_t.copy()]
    regrets = []
    f_star = f([0.7, 0.3])  # Approximate optimal value
    L = 2 + 0.1  # Lipschitz constant estimate: 2 from quadratic, 0.1 from L1
    
    for t in range(T):
        grad = grad_f(x_t)
        if adaptive:
            eta_t = min(eta, 1 / (L * np.sqrt(t + 1)))  # Adaptive step size
        else:
            eta_t = eta / np.sqrt(t + 1)  # O(1/sqrt(t)) step size
        x_t = mirror_descent_step(x_t, grad, eta_t)
        trajectory.append(x_t.copy())
        regrets.append(f(x_t) - f_star)
    
    return np.array(trajectory), np.array(regrets)

# Gradient Descent Algorithm for Comparison
def gradient_descent(x0, T, eta):
    """
    Standard Gradient Descent with projection onto the simplex.
    - Theoretical guarantee: Regret bound of O(sqrt(T)).
    """
    x_t = np.array(x0)
    trajectory = [x_t.copy()]
    regrets = []
    f_star = f([0.7, 0.3])
    
    for t in range(T):
        grad = grad_f(x_t)
        eta_t = eta / np.sqrt(t + 1)
        x_t = gradient_descent_step(x_t, grad, eta_t)
        trajectory.append(x_t.copy())
        regrets.append(f(x_t) - f_star)
    
    return np.array(trajectory), np.array(regrets)

# Theoretical regret bound
def theoretical_bound(T, R=1, L=2.1):
    """
    Theoretical regret bound for mirror descent: O(sqrt(T)).
    - R: Diameter of the feasible set.
    - L: Lipschitz constant of the gradient.
    """
    return R * L * np.sqrt(2 * np.arange(1, T + 1))

# Run the simulation
T = 1000  # Number of iterations
x0 = np.array([0.5, 0.5])  # Initial point on simplex
eta = 0.1  # Base step size

# Run mirror descent
md_traj, md_regrets = mirror_descent(x0, T, eta, adaptive=False)
md_traj_adaptive, md_regrets_adaptive = mirror_descent(x0, T, eta, adaptive=True)

# Run stochastic mirror descent
smd_traj, smd_regrets = stochastic_mirror_descent(x0, T, eta, noise_scale=0.1)

# Run gradient descent
gd_traj, gd_regrets = gradient_descent(x0, T, eta)

# Compute average iterates and their function values
md_avg = np.cumsum(md_traj[:-1], axis=0) / np.arange(1, T + 1)[:, None]
md_avg_f = np.array([f(x) for x in md_avg])
md_avg_adaptive = np.cumsum(md_traj_adaptive[:-1], axis=0) / np.arange(1, T + 1)[:, None]
md_avg_adaptive_f = np.array([f(x) for x in md_avg_adaptive])
smd_avg = np.cumsum(smd_traj[:-1], axis=0) / np.arange(1, T + 1)[:, None]
smd_avg_f = np.array([f(x) for x in smd_avg])
gd_avg = np.cumsum(gd_traj[:-1], axis=0) / np.arange(1, T + 1)[:, None]
gd_avg_f = np.array([f(x) for x in gd_avg])

# Theoretical bound
theory_bound = theoretical_bound(T)

# Visualization
plt.figure(figsize=(14, 10))

# Plot regret of iterates
plt.subplot(2, 1, 1)
plt.plot(md_regrets, label='Mirror Descent Regret', color='blue')
plt.plot(md_regrets_adaptive, label='Adaptive Mirror Descent Regret', color='green')
plt.plot(smd_regrets, label='Stochastic Mirror Descent Regret', color='purple')
plt.plot(gd_regrets, label='Gradient Descent Regret', color='red')
plt.plot(theory_bound, '--', label='Theoretical O(1/sqrt(T)) Bound', color='black')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Regret f(x_t) - f(x*)')
plt.title('Exploring the Complexity of Mirror Descent: Regret Analysis')
plt.legend()
plt.grid(True)

# Plot convergence of average iterates
plt.subplot(2, 1, 2)
plt.plot(md_avg_f - f([0.7, 0.3]), label='Mirror Descent Avg', color='blue')
plt.plot(md_avg_adaptive_f - f([0.7, 0.3]), label='Adaptive Mirror Descent Avg', color='green')
plt.plot(smd_avg_f - f([0.7, 0.3]), label='Stochastic Mirror Descent Avg', color='purple')
plt.plot(gd_avg_f - f([0.7, 0.3]), label='Gradient Descent Avg', color='red')
plt.plot(theory_bound / np.arange(1, T + 1), '--', label='Theoretical O(1/sqrt(T))', color='black')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('f(AVG(x_t)) - f(x*)')
plt.title('Convergence of Average Iterates')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final results
print(f"Final Mirror Descent Avg Regret: {md_avg_f[-1] - f([0.7, 0.3]):.6f}")
print(f"Final Adaptive Mirror Descent Avg Regret: {md_avg_adaptive_f[-1] - f([0.7, 0.3]):.6f}")
print(f"Final Stochastic Mirror Descent Avg Regret: {smd_avg_f[-1] - f([0.7, 0.3]):.6f}")
print(f"Final Gradient Descent Avg Regret: {gd_avg_f[-1] - f([0.7, 0.3]):.6f}")
print(f"Theoretical Bound at T={T}: {theory_bound[-1] / T:.6f}")
