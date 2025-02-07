"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from math import inf, sqrt

# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.X = np.empty(shape=(0, DOMAIN.shape[0]))
        self.f = np.empty(shape=0)
        self.v = np.empty(shape=0)

        self.var_f = 0.15**2
        self.var_v = 0.0001**2

        self.obj_model = GaussianProcessRegressor(
            kernel=0.5 * RBF(length_scale=1.0),
            alpha=self.var_f,
            normalize_y=False,
        )
        self.const_model = GaussianProcessRegressor(
            kernel=DotProduct(sigma_0=0) + sqrt(2) * RBF(length_scale=1.0),
            alpha=self.var_v,
            normalize_y=False,
        )

        self.max = -inf
        self.idx = 0

        self.acqui_mode = "ei"

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        if self.X.shape[0] == 0:
            return np.random.uniform(*DOMAIN[0])
        x_opt = self.optimize_acquisition_function()
        return x_opt

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(
                DOMAIN.shape[0]
            )
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        # TODO: Implement the acquisition function you want to optimize.

        x = np.atleast_2d(x)
        mu_f, std_f = self.obj_model.predict(x, return_std=True)
        mu_v, std_v = self.const_model.predict(x, return_std=True)
        mu_v = mu_v + SAFETY_THRESHOLD * np.ones_like(mu_v)

        if self.acqui_mode == "ucb":
            penalty = 8
            beta = 1
            return np.where(
                mu_v + 1.5 * std_v >= SAFETY_THRESHOLD,
                mu_f + std_f * beta - penalty,
                mu_f + std_f * beta,
            )

        elif self.acqui_mode == "pi":
            const_prob = norm.cdf(SAFETY_THRESHOLD, mu_v, std_v)
            pi = np.zeros_like(mu_f)
            for i in range(mu_f.shape[0]):
                pi[i] = 1 - norm.cdf(self.max, mu_f[i], std_f[i])
            return pi * const_prob

        elif self.acqui_mode == "ei":
            const_prob = norm.cdf(SAFETY_THRESHOLD, loc=mu_v, scale=std_v)
            xi = 0.0
            z_0 = mu_f - self.max - xi
            obj_ei = (z_0) * (norm.cdf(z_0 / std_f)) + std_f * norm.pdf(z_0 / std_f)
            return obj_ei * const_prob
            # if const_prob >= 0.97:
            #     return obj_ei
            # return 0.0

            # return np.where(
            #     mu_v + 1.5 * std_v <= SAFETY_THRESHOLD,
            #     obj_ei,
            #     np.zeros_like(mu_v),
            # )

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.X = np.vstack((self.X, x))
        self.f = np.hstack((self.f, f))
        self.v = np.hstack((self.v, v))

        self.obj_model.fit(X=self.X, y=self.f)
        self.const_model.fit(
            X=self.X, y=self.v - SAFETY_THRESHOLD * np.ones_like(self.v)
        )

        if v < SAFETY_THRESHOLD and f > self.max:
            self.max = f
            self.idx = self.f.shape[0] - 1

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        if self.X.shape[0] == 0:
            return np.random.uniform(*DOMAIN[0])
        return self.X[self.idx, 0]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---


def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return -np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), (
            f"The function next recommendation must return a numpy array of "
            f"shape (1, {DOMAIN.shape[0]})"
        )

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), (
        f"The function get solution must return a point within the"
        f"DOMAIN, {solution} returned instead"
    )

    # Compute regret
    regret = 0 - f(solution)

    print(
        f"Optimal value: 0\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n"
    )


if __name__ == "__main__":
    main()
