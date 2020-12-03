import lab as B
import pytest
from stheno import Measure, GP, Unique, Delta

from matrix import Woodbury
from .util import approx, IgnoreDenseWarning


@pytest.mark.parametrize("sample_truth", [True, False])
def test_blr(sample_truth):
    with IgnoreDenseWarning():
        def check_posterior(m, x):
            for p in [f_noisy, slope_noisy, intercept_noisy]:
                fdd = p(Unique(x.copy()))
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
        x = B.linspace(0, 10, 10_000)
        if sample_truth:
            y_obs = true_slope * x + true_intercept
        else:
            y_obs = f_noisy(x).sample()

        m = m | (m(f_noisy)(Unique(x.copy())), y_obs)
        check_posterior(m, x)

        m = m | (m(slope_noisy)(Unique(x0.copy())), true_slope)
        check_posterior(m, x)

        # Sample more and condition on `y` and the intercept.
        if sample_truth:
            y_obs = true_slope * x + true_intercept
        else:
            y_obs = m(f_noisy)(Unique(x.copy())).sample()

        m = m | (m(f_noisy)(Unique(x.copy())), y_obs)
        check_posterior(m, x)

        m = m | (m(intercept_noisy)(Unique(x0.copy())), true_intercept)
        check_posterior(m, x)

        if sample_truth:
            approx(m(slope_noisy)(x0).mean, true_slope, rtol=5e-2)
            approx(m(intercept_noisy)(x0).mean, true_intercept, rtol=5e-2)
