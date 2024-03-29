import jax.numpy as jnp
import lab.jax as B
import pytest
from stheno.jax import GP, Delta, Measure

from matrix import Woodbury

from .util import IgnoreDenseWarning, approx


@pytest.mark.parametrize("sample_truth", [True, False])
def test_blr(sample_truth):

    with IgnoreDenseWarning():

        def check_posterior(m, x):
            for p, noise in [
                (f_noisy, 0.2**2),
                (slope_noisy, 0.1**2),
                (intercept_noisy, 0.1**2),
            ]:
                fdd = p(x, noise)
                # Check that the posterior is of the right form and mimics the prior.
                assert isinstance(fdd.var, Woodbury)
                assert isinstance(m(fdd).var, Woodbury)
                approx(fdd.var.lr.left, m(fdd).var.lr.left)
                # Compute the Cholesky to check for positive definiteness.
                B.cholesky(fdd.var.lr.middle)

        # Construct the model.
        m = Measure()

        slope = GP(1, measure=m)
        intercept = GP(1, measure=m)
        f = slope * (lambda x: x) + intercept

        slope_noisy = slope + 0.1 * GP(Delta(), measure=m)
        intercept_noisy = intercept + 0.1 * GP(Delta(), measure=m)
        f_noisy = f + 0.2 * GP(Delta(), measure=m)

        # Sample slope and intercept.
        x0 = B.zeros(1, 1)
        true_slope = slope(x0).sample()[0, 0]
        true_intercept = intercept(x0).sample()[0, 0]

        # Sample and condition on `y` and the slope.
        x = B.linspace(0, 10, 1_000)
        if sample_truth:
            y_obs = true_slope * x + true_intercept
        else:
            y_obs = f_noisy(x).sample()

        m = m | (m(f)(x, 0.2**2), y_obs)
        check_posterior(m, x)

        m = m | (m(slope)(x0, 0.1**2), true_slope)
        check_posterior(m, x)

        # Sample more and condition on `y` and the intercept.
        if sample_truth:
            y_obs = true_slope * x + true_intercept
        else:
            y_obs = m(f)(x, 0.2**2).sample()

        m = m | (m(f)(x, 0.2**2), y_obs)
        check_posterior(m, x)

        m = m | (m(intercept)(x0, 0.1**2), true_intercept)
        check_posterior(m, x)

        if sample_truth:
            approx(m(slope_noisy)(x0).mean[0, 0], true_slope, rtol=5e-2)
            approx(m(intercept_noisy)(x0).mean[0, 0], true_intercept, rtol=5e-2)


def test_blr_jax_jit():
    with IgnoreDenseWarning():

        def build_model():
            prior = Measure()

            a = GP(1, measure=prior)
            b = GP(1, measure=prior)
            c = GP(1, measure=prior)
            f = a + b * (lambda x: x) + c * (lambda x: x**2)

            e1 = GP(0.1 * Delta(), measure=prior)
            e2 = GP(0.2 * Delta(), measure=prior)
            y1 = f + e1
            y2 = f + e2

            return prior, y1, y2

        @B.jit
        def posterior_marginals(x_obs, y_obs):
            prior, y1, y2 = build_model()
            post = prior | (y1(x_obs), y_obs)
            return post(y2)(x_obs).marginal_credible_bounds()

        # Sample some observations.
        _, y, _ = build_model()
        x_obs = B.linspace(jnp.float64, 0, 10, 10)
        y_obs = y(x_obs).sample()

        # Test that the JIT works and produces the correct result.
        approx(
            posterior_marginals(*B.to_numpy(x_obs, y_obs)),
            posterior_marginals(x_obs, y_obs),
            # Different computations can give different numerical behaviour.
            rtol=1e-5,
        )
