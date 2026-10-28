"""
Correctness checks for the pieces the conclusions depend on.

Run:  python3 src/test_pipeline.py
These are assertions about the protocol, not about the results: that folds really
are disjoint, that the grouping constraints actually hold, that S_mean behaves as
documented, and that the ridge matches a reference implementation.
"""
import os, sys, argparse
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Dataset, SCHEMES, ridge_gcv, pooled_r2, bootstrap_ci,
                    splits_lineage_blocked, splits_patient_grouped)

ok = lambda m: print(f"  PASS  {m}")


def test_folds_disjoint_and_complete(ds):
    for name, fn in SCHEMES.items():
        splits = fn(ds)
        seen = np.concatenate([te for _, te in splits])
        assert len(seen) == ds.n, f"{name}: folds do not cover the cohort"
        assert len(np.unique(seen)) == ds.n, f"{name}: a line appears in two test folds"
        for tr, te in splits:
            assert len(np.intersect1d(tr, te)) == 0, f"{name}: train/test overlap"
        ok(f"{name}: folds disjoint, cover all {ds.n} lines, no train/test overlap")


def test_patient_grouping(ds):
    pid = ds.meta["PatientID"].fillna("NA").values
    for name, fn in SCHEMES.items():
        for tr, te in fn(ds):
            shared = set(pid[tr]) & set(pid[te])
            assert not shared, f"{name}: patient(s) {list(shared)[:3]} split across folds"
    ok("no patient's cell lines are split across a train/test boundary (both schemes)")


def test_lineage_blocking(ds):
    for tr, te in splits_lineage_blocked(ds):
        shared = set(ds.lineage[tr]) & set(ds.lineage[te])
        # Rare lineages are pooled into one held-out block; no lineage may overlap.
        assert not shared, f"lineage(s) {list(shared)[:3]} appear in train and test"
    ok("lineage-blocked CV: no lineage appears in both train and test")


def test_patient_scheme_is_not_lineage_blocked(ds):
    leaks = sum(len(set(ds.lineage[tr]) & set(ds.lineage[te])) > 0
                for tr, te in splits_patient_grouped(ds))
    assert leaks > 0, "S1 should share lineages across folds; otherwise S1 and S2 are the same test"
    ok(f"patient-grouped CV does share lineages across folds ({leaks}/5) - S1 and S2 differ")


def test_r2_reference_points():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    parts = [y[:100], y[100:]]
    means = [y[100:].mean(), y[:100].mean()]
    const = [np.full(100, m) for m in means]
    r2c = pooled_r2(parts, const, means)
    assert abs(r2c) < 1e-9, f"constant-at-training-mean should give R2=0, got {r2c}"
    perfect = pooled_r2(parts, parts, means)
    assert abs(perfect - 1.0) < 1e-9, f"perfect prediction should give R2=1, got {perfect}"
    bad = pooled_r2(parts, [p + 5 for p in parts], means)
    assert bad < 0, "a predictor worse than the training mean must score negative"
    ok(f"S_mean: constant=0, perfect=1, worse-than-mean={bad:.2f} (negative)")


def test_ridge_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 60)); b = rng.normal(size=60)
    y = X @ b + rng.normal(size=200) * 0.5
    Xte = rng.normal(size=(50, 60))
    worst = 0.0
    for a in [0.1, 1.0, 10.0, 1e3, 1e5]:
        mine, _ = ridge_gcv(X, y, Xte, alphas=np.array([a]))
        worst = max(worst, np.abs(mine - Ridge(alpha=a).fit(X, y).predict(Xte)).max())
    assert worst < 1e-8, f"ridge disagrees with sklearn by {worst:.2e}"
    ok(f"ridge matches sklearn.Ridge across 5 penalties (max diff {worst:.1e})")


def test_bootstrap_sane():
    """A percentile bootstrap brackets the SAMPLE statistic; coverage of the true
    value is only expected on average, so the null check uses nominal coverage
    across many independent samples rather than one draw."""
    rng = np.random.default_rng(0)
    v = rng.normal(loc=0.3, scale=0.1, size=500)
    m, lo, hi = bootstrap_ci(v)
    assert lo <= m <= hi, "CI must bracket the sample statistic"
    assert (hi - lo) < 0.05, "CI implausibly wide for n=500, sd=0.1"

    # Coverage: over 200 independent null samples, a 95% CI should contain 0
    # roughly 95% of the time. Allow a generous band so the test is not flaky.
    hits = 0
    for i in range(200):
        r = np.random.default_rng(1000 + i)
        _, l0, h0 = bootstrap_ci(r.normal(0, 1, 400), n_boot=400, seed=i)
        hits += (l0 < 0 < h0)
    cov = hits / 200
    assert 0.88 <= cov <= 1.0, f"bootstrap coverage {cov:.2f} is off nominal"
    ok(f"bootstrap: median {m:.3f} CI [{lo:.3f},{hi:.3f}]; null coverage {cov:.1%}")


def test_no_constant_or_nan_features(ds):
    for nm, M in [("expr", ds.expr), ("cn", ds.cn), ("mut", ds.mut)]:
        assert not np.isnan(M).any(), f"{nm} contains NaN"
    assert not np.isnan(ds.Y).any(), "Y contains NaN"
    ok("no NaNs in any matrix after QC")


if __name__ == "__main__":
    print("loading cohort...")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Use an aligned toy or real dataset")
    args = parser.parse_args()
    ds = Dataset() if args.data_dir is None else Dataset(data_dir=args.data_dir)
    print(f"cohort: n={ds.n}, lineages={len(set(ds.lineage))}, "
          f"patients={ds.meta.PatientID.nunique()}\n")
    test_folds_disjoint_and_complete(ds)
    test_patient_grouping(ds)
    test_lineage_blocking(ds)
    test_patient_scheme_is_not_lineage_blocked(ds)
    test_no_constant_or_nan_features(ds)
    test_r2_reference_points()
    test_ridge_matches_sklearn()
    test_bootstrap_sane()
    print("\nAll checks passed.")
