# Deviations from the preregistered design

Each entry records what changed, when, and whether any results had been seen.

---

### D1 — Feature-budget sensitivity grid narrowed
- **Design said:** §11.3 sensitivity at top-k ∈ {100, 500, 2000}.
- **Changed to:** primary k = 300; sensitivity at k ∈ {100, 600}, and the sensitivity
  runs cover only arms A and E on the selective stratum.
- **Reason:** compute. Ridge cost scales roughly with k², and k = 2000 across six arms
  and ten fold-by-scheme combinations would have run for many hours on 2 CPU cores.
- **When:** before any model was fit. **Results seen at the time:** none.
- **Effect on inference:** none for the primary endpoint, which is defined at fixed k.
  It does narrow the range over which feature-budget robustness is demonstrated, and that
  limitation is stated in the report.

### D2 — Selective-gene cap never binds
- **Design said:** §7 selective genes capped at N = 1000 by descending variance.
- **Observed:** only **581** genes meet the selective criteria, so the cap is inactive and
  the full selective stratum is used.
- **When:** at target-set construction, before fitting. **Results seen:** none.
- **Effect:** none. Recorded because the reported N differs from the preregistered ceiling.

### D3 — Alpha selection uses GCV rather than exact leave-one-out
- **Design said:** §5 "alpha chosen by inner CV from a fixed grid."
- **Implemented as:** generalized cross-validation on one SVD per fit.
- **Reason:** exact LOO per fit was too slow at this scale. Verified against
  `sklearn.linear_model.Ridge` at fixed alpha (agreement to 3e-14) and against
  `RidgeCV` for alpha selection, where the two chose **adjacent** grid points
  (1.00 vs 1.78 on the validation problem; grid spacing is 1.78x).
- **When:** before any model was fit. **Results seen:** none.
- **Effect on inference:** negligible — the comparison between arms is paired and every
  arm uses the identical selection rule.
