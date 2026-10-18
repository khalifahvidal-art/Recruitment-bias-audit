# Synthetic recruitment intervention experiment

Version 1.0.0, prepared 16 October 2026 for Khalifah Vidal.

This is entirely synthetic research. CaribBank is fictional. No real applicant data are included. This package is not suitable for real hiring or compliance certification.

## Reproduction

Use Python 3.12 and a separate environment, install `requirements.txt`, then run:

```
python run_experiment.py --seeds 30
python plot_results.py
```

Use `--out alternative_outputs` to preserve another numerical run. The plotting script reads `outputs` by default. Consult `outputs/environment.json` for executed versions. Numeric dependencies and solver changes may alter results. The private Fairlearn `_pmf_predict` method is used only for expected-mixture training diagnostics, so Fairlearn is pinned.

## Design

Thirty seeds (0–29), fifteen scenarios, joint group/qualification stratified 70/30 splits. Nine learned reference conditions plus an oracle; five learned conditions plus an oracle per other scenario. Within each scenario and seed, all methods share data and split. Group is excluded from scoring features. Conditions vary training labels, weights or TPR constraints. Bounds are group-versus-pooled training conditions, not test guarantees.

One factor changes at a time: suppression, proxy separation, sample size, group proportion, base-rate shift or symmetric review noise. This does not estimate factorial interactions. The oracle knows the generator probability but not realised Bernoulli labels. It is an expected-risk reference, not a hard upper bound for finite samples.

Bootstrap intervals resample seed-level paired differences 10,000 times (seed 20261016). They describe simulation uncertainty, not population uncertainty for Saint Lucia. No adjustment for simultaneous inference is applied.

## Verification

`verification.json` records independent count-based metric checks and exact continuity of the initial ten reference-run metrics against the original study outputs. All original ten-seed accuracy, TPR-gap and FPR-gap results were reproduced exactly. The seed-0 generator also matched. `verify_experiment.py` includes the independent arithmetic checks and continuity checks; the latter require original reference files which are included under `original_reference`.

## Interpretation

The manuscript accompanying this package is the first project submission, focused on rationale, framework and methodology. Full results and discussion belong to submission 2. Summary values can be read from `outputs/summary.csv`; paired effects from `outputs/paired_intervals.csv`. Plot error bars in the reference plot are sample standard deviations. The sensitivity plot uses paired bootstrap intervals.

Identical realised .05 and .10 reviewed-condition metrics do not imply identical mixture probabilities. Expected training deviations are below .05 in the reference runs; the constraint is slack there, while solver details can still change a mixture slightly. The .02 incremental gap interval crosses zero; this is not evidence of equivalence. The .05 incremental gap is positive under the reference generator, but sensitivity scenarios must be considered before generalising.

## Release and authorship

This repository publishes experiment package version 1.0.0. No Zenodo DOI has been issued. The complete numerical outputs, figures and verification record are included. No open-source licence is granted by this deposit. A GitHub release tag is to be created for the published snapshot.

Generative AI assisted manuscript organisation/editing, code adaptation, source identification and figure preparation. The author must review all content, understand each design choice and make the programme-required declaration. No ethics approval or participant findings are claimed.
