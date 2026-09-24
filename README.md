# Quantum Optimal Transport for Cross-Site Domain Adaptation in Railroad Monitoring

Code and per-fold results for the study of the same name, submitted to the ASCE
*Journal of Infrastructure Systems*.

The problem is cross-site transfer. A classifier is fitted at one railroad test
site to predict, from ground-penetrating radar ballast indices, whether a track
segment carries a 62-ft profile deviation above an intervention threshold. The
question is whether that classifier can be carried to a second site where little
labeled geometry is available, by aligning the two sites' feature distributions
with an optimal transport map.

Two families of transport are compared on identical folds, readouts and
classifiers, so that the transport step is the only thing that differs.

- **QOT**, the Bures metric transport map applied in closed form to a segment
  density matrix.
- **COT**, entropic Sinkhorn transport, applied both to the raw feature vector
  and to the vectorized density matrix.

Each transport is run against a control that is the same pipeline with the
transport step removed and nothing else changed.

## What is here, and what is not

The analysis code and the per-fold result files are in this repository.

The ground-penetrating radar indices and the track geometry records are not.
They were provided by a third party and are proprietary. Requests for them go to
the data owner. The scripts that need them say so and name the columns they
expect, so the pipeline can be run against equivalent data from elsewhere.

The result CSVs are derived outputs. They hold one row per fold and method with
balanced accuracy, precision and recall. They carry no raw radar or geometry
measurement. Every table and figure in the paper regenerates from them without
any access to the withheld inputs.

## Layout

```
run_everything.py          full experiment, amplitude encoding
run_angle.py               the same experiment, angle encoding
run_uncond_classical.py    the unconditional Sinkhorn comparators
rerun_sweeps8.py           sensitivity sweeps at eight starting positions
rerun_perm8.py             label permutation test at eight starting positions
test_magnitude.py          does discarding absolute severity help transfer
leftrail.py                why transport gives no gain on the left rail

make_tables_3_4.py         Tables 3 and 4 from the result CSVs
make_tables_5_6.py         Tables 5 and 6 from the result CSVs
make_sensitivity_report.py recomputes every number in the Sensitivity subsection
make_fig1.py               Figure 1, ballast index distributions
make_fig2.py               Figure 2, the pipeline diagram
make_fig_labeleff.py       Figure 3, label efficiency

qot_results/               amplitude encoding, per-fold CSVs and run report
qot_angle_results/         angle encoding, per-fold CSVs and run report
```

## Reproducing the tables and figures

This needs nothing but the repository.

```
pip install numpy pandas scipy scikit-learn matplotlib

python3 make_tables_3_4.py          # tables/table3_*.tex, tables/table4_*.tex
python3 make_tables_5_6.py          # tables/table5_*.tex, tables/table6_*.tex
python3 make_fig1.py                # figures/fig_eda_stacked_grey.pdf
python3 make_fig2.py                # figures/fig_pipeline.pdf
python3 make_fig_labeleff.py        # figures/fig_labeleff.pdf
python3 make_sensitivity_report.py  # qot_results/SENSITIVITY_REPORT.txt
```

The table scripts write LaTeX table bodies, the rows that sit between
`\midrule` and `\bottomrule`.

`make_sensitivity_report.py` recomputes every numeric claim in the Sensitivity
subsection from the result CSVs and prints `MATCH` or `DIFFERS` against the
value printed in the paper. One of its checks, the correlation between ballast
indices and the per-rail label, needs the withheld input files and is skipped
with a message when they are absent. Everything else runs.

## Re-running the experiment from raw data

This needs the withheld input files. Order matters, because the later scripts
import the earlier ones to reuse the fold construction rather than
reimplementing it.

```
OMP_NUM_THREADS=1 python3 run_everything.py       --data /path/to/inputs
OMP_NUM_THREADS=1 python3 run_angle.py            --data /path/to/inputs
OMP_NUM_THREADS=1 python3 run_uncond_classical.py --data /path/to/inputs --enc Amp
OMP_NUM_THREADS=1 python3 run_uncond_classical.py --data /path/to/inputs --enc Ang
OMP_NUM_THREADS=1 python3 rerun_sweeps8.py        --data /path/to/inputs --enc Amp
OMP_NUM_THREADS=1 python3 rerun_sweeps8.py        --data /path/to/inputs --enc Ang
OMP_NUM_THREADS=1 python3 rerun_perm8.py          --data /path/to/inputs --enc Amp
OMP_NUM_THREADS=1 python3 rerun_perm8.py          --data /path/to/inputs --enc Ang
OMP_NUM_THREADS=1 python3 test_magnitude.py       --data /path/to/inputs
```

Each of the two main scripts takes hours. Both accept `--quick` for a smoke
test and `--stage <name>` to run one stage at a time. `OMP_NUM_THREADS=1` is
set because the linear algebra is small and thread contention makes it slower,
not faster.

### What the input files must contain

One file per site, named `cleaned_HTL.csv`, `cleaned_PTT.csv`, `cleaned_RTT.csv`
and `cleaned_WRM.csv`. One row per track geometry record, with the segment's
ballast indices repeated on every record belonging to that segment.

| Column | Meaning |
| --- | --- |
| `run_id` | Geometry run identifier. One run per site is retained, the one closest in date to that site's radar survey. |
| `BMP`, `EMP` | Begin and end milepost of the segment the record falls in. These two together identify a segment. |
| `LProf62`, `RProf62` | Left and right rail 62-ft chord profile deviation, in inches. |
| `BFI_L_Cat`, `BFI_C_Cat`, `BFI_R_Cat` | Ballast Fouling Index, ordinal 1 to 5, at the left rail, track center and right rail. |
| `BTI_L`, `BTI_C`, `BTI_R` | Ballast Thickness Index, ordinal 1 to 5. |
| `MLI_L`, `MLI_C`, `MLI_R` | Moisture Likelihood Index, ordinal 1 to 3. |
| `LRI_L`, `LRI_C`, `LRI_R` | Layer Roughness Index, ordinal 1 to 3. |

Segment construction, identical in every script:

- Keep only the rows whose `run_id` matches the retained run for that site.
- Group by `(BMP, EMP)`. That group is a segment.
- Drop segments holding fewer than ten geometry records, since the label is a
  maximum over those records and is unreliable when few are available.
- The eight ballast features take a single value across a segment, so the first
  value in the group is used.
- A segment is labeled a defect on a rail when the largest absolute 62-ft
  profile deviation on that rail anywhere in the segment exceeds 0.4 in
  (10 mm).

The retained `run_id` values are hardcoded at the top of `run_everything.py`
and would need changing for a different dataset.

## The result files

A fold is one combination of source site, target site, starting position and
labeled percentage. The evaluation half of the target site is started at eight
positions spaced evenly along the site, which gives eight folds per site pair
and labeled percentage. The evaluation half is fixed and never receives labels,
and a 200 ft buffer separates it from the labeled stretch.

The code and the CSVs call that starting position a **rotation**, and the `Rot`
column holds it. The paper calls the same thing a **fold**. The column name is
left as it is so that the published CSVs are unchanged from the run.

Columns in the main result files:

| Column | Meaning |
| --- | --- |
| `Side` | `Left` or `Right` rail. |
| `Arc` | `contig` or `random`, how the labeled target segments were drawn. |
| `Source`, `Target` | The two site codes of the directed pair. |
| `Rot` | Starting position, 0 to 7. Called a fold in the paper. |
| `AdaptFrac` | Fraction of the target site made available for adaptation. |
| `ArcOK` | Whether the labeled stretch held both classes on this fold. |
| `nEval`, `nEvalPos` | Evaluation segments, and how many are defects. |
| `nArc`, `nArcPos` | Labeled target segments, and how many are defects. |
| `Method` | Internal method name, see the mapping above. |
| `Uses` | What the method reads from the target site. |
| `K` | PCA components chosen on this fold, for the PCA readout only. |
| `BAC`, `Precision`, `Recall` | Scores on the evaluation half. |

F1 is not stored. It is computed as 2PR/(P+R) per fold where a table needs it.

| File | Holds |
| --- | --- |
| `main_<Side>_contig*.csv` | The main grid. Labeled segments drawn as a contiguous stretch. This backs Tables 3 to 6. |
| `main_<Side>_random*.csv` | The same grid with labeled segments drawn at random across the site. This backs the label-construction check. |
| `uncond_classical*.csv` | COT-u and COT-rho-u, the unconditional Sinkhorn comparators. |
| `sweeps8_*.csv` | The Bures floor and the Sinkhorn regularization, at eight starting positions. |
| `alpha8_Ang.csv` | The angle-scaling sweep. |
| `perm8_*.csv` | The label permutation test. |
| `bootstrap.csv` | Moving-block bootstrap intervals, one per site pair. |
| `magnitude_test.csv` | The three arms of the magnitude test. |
| `REPORT.txt` | Everything the run printed. Stages superseded by the eight-position reruns are marked. |
| `SENSITIVITY_REPORT.txt` | Generated by `make_sensitivity_report.py`. |

Internal method names map to the names used in the paper as follows.

| In the CSVs | In the paper |
| --- | --- |
| `QOTu-HSsrc` | QOT-u, HS-2 readout |
| `QOTu-HS4` | QOT-u, HS-4 readout |
| `QOTu-PCA` | QOT-u, PCA readout |
| `QOTc-*` | QOT-c |
| `SDMu-*` | COT-rho-u |
| `SDM-*` | COT-rho-c |
| `none-*` | no transport |
| `COT-u` | COT-u |
| `COT` | COT-c |
| `NoAdapt` | no adaptation |
| `TargetOnly` | target only |
| `Source+Target` | source + target |

Angle-encoding method names carry `-Ang-` in the middle, for example
`QOTu-Ang-HSsrc`. The loader in `make_tables_3_4.py` strips it.

`HSsrc` is the HS-2 readout, two prototypes built from the source class means
and nothing from the target. `HS4` is HS-4, which adds two prototypes from the
labeled target segments.

## A note on superseded runs

The sweep and permutation stages inside `run_everything.py` and `run_angle.py`
run at four starting positions. The main results use eight. Those stages are
superseded by `rerun_sweeps8.py` and `rerun_perm8.py`, and the reported
sensitivity results come from the eight-position reruns. The superseded CSVs are
not included here, and the sections of `REPORT.txt` they came from are marked.

## Requirements

Python 3.9 or later. `requirements.txt` gives lower bounds rather than exact
pins. Small differences in the third decimal are possible across library
versions, since the Sinkhorn solver and the eigendecomposition inside the Bures
map are both iterative. The per-fold CSVs in this repository are the run the
paper reports.

## Contact

Dengimowei David Alabintei, Department of Civil and Environmental Engineering,
University of Maryland, College Park.
alabinte@umd.edu, davidalabintei97@gmail.com

Requests for the ground-penetrating radar and track geometry data go to the
data owner, not to this repository.

## License

MIT. See `LICENSE`.
