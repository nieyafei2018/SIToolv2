# Sea Ice Evaluation Tool (SITool) v2.0

## 1. Overview

SITool (v2.0) evolved from version 1.0 (Lin et al., 2021; https://github.com/XiaLinUCL/Sea-Ice-Evaluation-Tool) and has undergone substantial modifications in its **code structure**, **workflow**, **metrics**, and **visualization** of results. The scientific objective of the current version is to conduct a **process-oriented** evaluation and analysis of large-scale polar sea ice simulations.

### 1.1 Typical Use Cases

- Comparative evaluation between CMIP/reanalysis products and observed sea ice products
- Quickly diagnose model performance or differences between experiments for sea ice modelers
- Correlation analysis between simulation errors of different variables

### 1.2 Supported Evaluation Modules

| Module      | Variable Theme               | Primary Processing Frequency | Typical Outputs                                                                                              |
| ----------- | ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `SIconc`    | Sea Ice Concentration        | monthly                      | SIC time series/anomalies/spatial maps, IIEE, heatmaps                                                       |
| `SIdrift`   | Sea Ice Drift (u/v)          | monthly                      | Drift vector maps, MKE maps, vector-correlation maps                                                         |
| `SIthick`   | Sea Ice Thickness            | monthly                      | Thickness/volume diagnostics in both original and obs-matched coverage (time series, maps, trends, scalars)  |
| `SNdepth`   | Snow Depth                   | monthly                      | Snow depth/volume diagnostics in both original and obs-matched coverage (time series, maps, trends, scalars) |
| `SICB`      | Sea Ice Concentration Budget | daily                        | Seasonal budget-component maps, comparison maps, time series                                                 |
| `SIMbudget` | Sea Ice Mass Budget          | monthly                      | Model-only seasonal budget maps, seasonal cycle lines, regional seasonal scalar tables                       |
| `SNMbudget` | Snow Mass Budget             | monthly                      | Model-only seasonal budget maps, seasonal cycle lines, regional seasonal scalar tables                       |
| `SItrans`   | Seasonal Transition Dates    | daily                        | Advance/retreat climatology, variability, trends, IIEE, relationship/regional diagnostics                    |

## 2. Main Workflow at a Glance

The user is responsible for preparing data, filling in the recipe, and analyzing the HTML output, while SITool v2 handles data checking, preprocessing, metric calculation, and report generation.

```text
cases/recipe_<case>.yml
        |
        v
RecipeReader (validate paths, variable names, time coverage, temporal resolution)
        |
        v
DataPreprocessor (seldate / monmean / remap to a common evaluation grid)
        |
        v
SeaIceMetrics (compute single model & inter-model metrics per module)
        |
        +--> metrics/*.nc cache (separate by module + hemisphere)
        |
        v
plot_figs (batch plotting)
        |
        v
report.generate_html_report -> Output/summary_report.html
```

## 3. Installation

### Prerequisites

- Supported operating systems: Linux and macOS
- Python 3.11.5 (pinned in `environment.yml`)
- `conda`
- Climate Data Operator (`cdo`, version 2.4.0+)

```bash
git clone https://github.com/nieyafei2018/SIToolv2
cd SIToolv2
conda env create -f environment.yml
conda activate sitoolv2
```

This project heavily uses `python-cdo` to call CDO CLI tools (for example `remapbil`, `monmean`, `seldate`).

- Ensure `cdo` is executable on your system
- If `cdo` is not installed (Debian/Ubuntu), install it with:

```bash
sudo apt install cdo
```

- Recommended quick check:

```bash
cdo -V
```

Note: `environment.yml` installs `python-cdo`, but runtime still depends on a system-level CDO binary.

## 4. Quick Start

### 4.1 Prepare a Recipe

Create or copy a file under `cases/`:

```text
cases/recipe_<case_name>.yml
```

Available example recipes in this repository include:

- `cases/recipe_highres.yml`
- `cases/recipe_simbudget.yml`
- `cases/recipe_snmbudget.yml`

In `recipe`, at least these two root directories must be configured correctly:

- `SIToolv2_RefData_path`: root directory of reference data
- `model_data_path`: root directory of model data

### 4.2 `SIToolv2_RefData` Directory and File Layout

`SIToolv2_RefData_path` in recipe should point to a directory with the following structure.
This is the full layout corresponding to the built-in offline reference-data preparation workflow
(`cases/refdata_prep_default.yml`) and the example recipes in this repository:

```text
SIToolv2_RefData/
|-- Auxiliary/
|   |-- ETOPO_2022_v1_60s_N90W180_surface.nc
|   |-- NSIDC0771_CellArea_PS_N25km_v1.0.nc
|   |-- NSIDC0771_CellArea_PS_S25km_v1.0.nc
|   |-- NSIDC0771_LatLon_PS_N25km_v1.0.nc
|   |-- NSIDC0771_LatLon_PS_S25km_v1.0.nc
|   |-- NSIDC-0780_SeaIceRegions_PS-N3.125km_v1.0.nc
|   `-- NSIDC-0780_SeaIceRegions_PS-S3.125km_v1.0.nc
|-- SIconc/
|   |-- NSIDC_CDR_siconc_daily_nh_19790101-20241231.nc
|   |-- NSIDC_CDR_siconc_daily_sh_19790101-20241231.nc
|   |-- OSI-450_siconc_daily_nh_19790101-20231231.nc
|   `-- OSI-450_siconc_daily_sh_19790101-20231231.nc
|-- SIdrift/
|   |-- NSIDC_PolarPathfinder_sidrift_daily_nh_19790101-20231231.nc
|   |-- NSIDC_PolarPathfinder_sidrift_daily_sh_19790101-20231231.nc
|   |-- OSI-455_sidrift_daily_nh_19910101-20201231.nc
|   `-- OSI-455_sidrift_daily_sh_19910101-20201231.nc
|-- SIthick/
|   |-- PIOMAS_nh_sithick_mon_197901-202312.nc
|   |-- TOPAZ4b_nh_sithick_mon_199101-202312.nc
|   |-- GIOMAS_sh_sithick_mon_197901-202312.nc
|   |-- CTOH_sithick_monthly_nh_199404-202306.nc
|   |-- CTOH_sithick_monthly_sh_199404-202306.nc
|   `-- CSBD_ubristol_cs2_sit_nh_80km_v1p7_monthly_201011-202007.nc
`-- SNdepth/
    |-- SnowModel-LG_nh_snod_ERA5_v01_19800801-20210731.nc
    |-- SnowModel-LG_nh_snod_MERRA2_v01_19800801-20210731.nc
    |-- CTOH_sndepth_monthly_nh_199410-202304.nc
    |-- CTOH_sndepth_monthly_sh_199404-202306.nc
    `-- MPMR_SNdepth_sh_20020601-20200531_monmean.nc
```

Notes:

- Runtime file discovery is module-based:
  - `SIconc` and `SItrans` read files from `SIToolv2_RefData/SIconc/`
  - `SIdrift` reads from `SIToolv2_RefData/SIdrift/`
  - `SIthick` reads from `SIToolv2_RefData/SIthick/`
  - `SNdepth` reads from `SIToolv2_RefData/SNdepth/`
  - `SICB` reads both `SIconc/` and `SIdrift/`
- Recipe entries such as `ref_nh`, `ref_sh`, `ref_nh_sic` use **basename only**
  (for example `NSIDC_CDR_siconc_daily_nh_19790101-20241231.nc`), not relative paths.
- You only need files that are explicitly referenced by your recipe; extra files in the same module directory are allowed.

### 4.3 Run

```bash
# Run all enabled modules in recipe_highres.yml
python main.py highres

# Run specific modules
python main.py highres --modules SIconc SIdrift

# Force recomputation (ignore cache) and print more detailed logs
python main.py highres --recalculate --log-level DEBUG
```

### 4.4 CLI Arguments

```text
python main.py <case_name> [--modules ...] [--log-level ...] [--recalculate] [-j N] [--keep-staging] [--rotate FILE,VAR]
```

- `case_name`: maps to `cases/recipe_<case_name>.yml`
- `--modules`: choose from `SIconc SIdrift SIthick SNdepth SICB SIMbudget SNMbudget SItrans`
- `--log-level`: `DEBUG|INFO|WARNING|ERROR`
- `--recalculate`: ignore cache and recompute metrics
- `-j/--jobs`: total parallel workers (the scheduler will cap to safe values automatically)
- `--keep-staging`: keep per-run staging files under `cases/<case>/metrics/_staging`
- `--rotate`: currently kept for backward compatibility; no longer an active control in the main workflow

> Note: with `--jobs=1`, modules run sequentially (hemisphere-major); larger values enable dynamic task scheduling.

## 5. Output Directory Layout

After one run (`<case>` example: `highres`):

```text
cases/<case>/
|-- <case>.log
|-- cpu_realtime_efficiency_latest.png            # real CPU utilization figure (generated when jobs > 1)
|-- Processed/
|   |-- nh/
|   `-- sh/
|-- Output/
|   |-- summary_report.html
|   |-- report_download.js                        # browser-side xlsx export helper
|   |-- report_cross_module.js                    # cross-module explorer logic
|   |-- cross_module_metrics.json                 # flattened scalar payload
|   |-- .metric_tables.pkl                        # internal snapshot for incremental report refresh
|   |-- nh/
|   |   |-- SIconc/
|   |   |-- SIdrift/
|   |   |-- SIthick/
|   |   |-- SNdepth/
|   |   |-- SICB/
|   |   |-- SIMbudget/
|   |   |-- SNMbudget/
|   |   `-- SItrans/
|   `-- sh/
|       `-- ...
`-- metrics/
    |-- nh_SIconc_metrics.nc
    |-- nh_SIdrift_metrics.nc
    |-- ...
    `-- <case>_all_metrics.nc                     # optional (enabled by SITOOL_BUILD_UNIFIED_CACHE=1)
```

Notes:

- `Processed/`: intermediate files after regridding
- `Output/<hms>/<module>/`: PNG figures
- `Output/summary_report.html`: consolidated interactive report
- `Output/cross_module_metrics.json`: scalar-table records for cross-module analysis
- `<case>.log`: complete runtime log
- `metrics/`: module cache files

## 6. How to Read `summary_report.html` Scientifically

After running SITool v2, all figures and statistical metrics are compiled into the file `cases/<case>/Output/summary_report.html` for easy viewing and analysis.

### 6.1 Page Structure

- Top-level hemisphere tabs: `Arctic (NH)` and `Antarctic (SH)`
- Left sidebar module navigation for the active hemisphere
- Optional `Geographical sector` panel (regional mask map)
- Optional `Cross-Module` panel (interactive scalar-vs-scalar explorer)

### 6.2 Three Interpretation Axes in Tables

1. **Coverage axis**
   
   - `Original Coverage`: each dataset uses its own valid cells
   - `Obs-Matched Coverage`: comparisons restricted to common-valid observation coverage

2. **View axis**
   
   - `Raw Values`: absolute climatology/variability/trend values
   - `Differences`: physical-difference diagnostics relative to `obs1` (keeps native units in each column; smaller absolute value means closer to `obs1`)
   - In `Differences` tables: `obs1` row is `0`, `obs2` row is `|obs2-obs1|`, model rows are `|model-obs1|`

3. **Domain axis**
   
   - region (`All` + sea sectors)
   - period/season/phase tabs depending on module

Recommended reading order: `Raw Values` first (physical realism), `Differences` second (model skill).

### 6.3 Module-Specific Scalar Semantics

- `SIconc`
  
  - Region tabs + period tabs (`Annual`, `March`, `September` in report UI)
  - `Raw`: SIE/SIA/MIZ/PIA mean, trend, detrended std
  - `Differences`: physical differences against `obs1` for mean/trend/variability/IIEE metrics
  - `*` on trend means `p < 0.05`

- `SIdrift`
  
  - `Raw`: speed climatology and anomaly variability/trend diagnostics
  - `Differences`: physical drift-error diagnostics against `obs1`
  - Compare original and obs-matched coverage before conclusion

- `SIthick` / `SNdepth`
  
  - `Raw`: field mean/std/trend plus volume mean/std/trend
  - `Differences`: physical thickness/snow diagnostics against `obs1`
  - Coverage sensitivity is scientifically critical

- `SICB`
  
  - Seasonal tabs (`Spring/Summer/Autumn/Winter`)
  - Columns: `dadt`, `adv`, `div`, `res` + each component's `% of dadt`
  - If `dadt` is near zero, percentage columns can be numerically unstable

- `SIMbudget` / `SNMbudget`
  
  - Model-only regional seasonal tables (`Gt/season`), no observation baseline by design

- `SItrans`
  
  - Phase tabs (`Advance`, `Retreat`) with Raw and Differences views
  - Raw includes mean date, variability, trend, valid-year and significance summaries
  - Differences are physical/model-skill diagnostics against `obs1` (native units per column)

### 6.4 Figure Semantics in HTML

- `_raw` and `_diff` figure suffixes follow table view semantics
- `Differences` tables keep physical units; only heatmap figures use ratio-to-observational-uncertainty (`|E_model| / |E_obs2|`) for model rows
- In heatmaps, the `obs2` row is displayed as physical baseline values (`|E_obs2|`), while model rows are ratio-colored
- For heatmaps, correlation-like metrics are internally converted to error form (`1 - |r|`) before ratioing
- Trend interpretation should always be paired with significance diagnostics

### 6.5 Cross-Module Explorer

The cross-module panel uses flattened scalar-table records (`cross_module_metrics.json`) and provides:

- Axis selection chain: `Module -> Coverage -> View -> Region -> Domain -> Metric`
- Model include/exclude checkboxes
- Optional `obs1` / `obs2` inclusion
- Stats: sample size `n`, Pearson `r`, Spearman `rho`, OLS regression equation, permutation p-value (default 1200 permutations), quadrant counts, leave-one-out max `Delta r`
- Export tools: paired-value CSV and 300-DPI PNG scatter

Scientific recommendation:

- Keep X/Y on comparable temporal domains (`season` vs `season`, `month` vs `month`)
- Avoid over-interpreting p-values for very small `n`
- Prefer convergent evidence across maps, scalar tables, and cross-module relationships

### 6.6 Practical Reading Workflow

For day-to-day model diagnosis, the following sequence is recommended:

1. Start from `All` region + `Raw Values` to check physical magnitude and sign
2. Switch to `Differences` to identify dominant model-vs-observation error terms
3. Move to sector tabs to test whether conclusions are regionally robust
4. Use `Cross-Module` last to check whether cross-variable relationships support the same story

This order helps avoid over-interpreting a single metric, a single panel, or a single statistical test.

## Contact

- Author: Yafei Nie
- Email: nieyafei@sml-zhuhai.cn
