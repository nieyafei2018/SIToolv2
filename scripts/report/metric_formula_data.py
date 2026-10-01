# -*- coding: utf-8 -*-
"""Metric naming and formula content for HTML report page."""

from typing import Dict, List


METRIC_FORMULA_PAGE: Dict[str, object] = {
    'title': 'Metric Glossary & Formulas',
    'intro': [
        'This page summarizes naming conventions and metric formulas used in tables and heatmaps.',
        'All formulas describe the current implementation and are aligned with report semantics.',
    ],
    'sections': [
        {
            'title': '1. Naming Convention',
            'paragraphs': [
                'Metric names follow a compact pattern: variable + statistic + scope + operator.',
            ],
            'formula_blocks': [
                {
                    'name': 'Canonical Pattern',
                    'equation': 'metric_code = module.variable.stat.scope.operator',
                    'notes': [
                        'module: SIconc / SIdrift / SIthick / SNdepth / SICB / SItrans ...',
                        'stat: Mean, Trend, DStd, RMSE, Corr ...',
                        'scope: TS (time series), Map (spatial map), Seasonal, Phase ...',
                        'operator: Raw, AbsDiff, Bias, Ratio ...',
                    ],
                },
            ],
        },
        {
            'title': '2. Core Statistics',
            'paragraphs': [
                'Table and heatmap statistics are generated from module-level metrics relative to obs1 baseline when applicable.',
            ],
            'formula_blocks': [
                {
                    'name': 'Raw Value',
                    'equation': 'Raw = x',
                    'notes': ['Direct metric value without obs-reference differencing.'],
                },
                {
                    'name': 'Absolute Difference',
                    'equation': 'AbsDiff = |x_model - x_obs1|',
                    'notes': ['Used in many model-vs-obs1 summary tables and heatmaps.'],
                },
                {
                    'name': 'Signed Bias',
                    'equation': 'Bias = x_model - x_obs1',
                    'notes': ['Retains sign information (positive/negative offset).'],
                },
            ],
        },
        {
            'title': '3. Trend and Detrended STD',
            'paragraphs': [
                'STD fields in current core modules are detrended STD unless explicitly stated otherwise.',
            ],
            'formula_blocks': [
                {
                    'name': 'Linear Trend Model',
                    'equation': 'y_t = a*t + b + eps_t',
                    'notes': [
                        'a is the linear slope estimated from valid samples.',
                        'Reported trend uses a time-scale factor based on metric units (for example per decade).',
                    ],
                },
                {
                    'name': 'Detrended Standard Deviation',
                    'equation': 'sigma_det = std( y_t - (a_hat*t + b_hat) )',
                    'notes': [
                        'Detrending is applied before standard deviation is computed.',
                        'Map-level detrended STD applies the same equation at each grid cell.',
                    ],
                },
            ],
        },
        {
            'title': '4. Heatmap Ratio Semantics',
            'paragraphs': [
                'For normalized heatmaps, values compare model error magnitude against obs2 uncertainty magnitude.',
            ],
            'formula_blocks': [
                {
                    'name': 'Ratio Display',
                    'equation': 'Ratio = |E_model| / max(|E_obs2|, eps)',
                    'notes': [
                        'E_model: model-to-obs1 error for one metric.',
                        'E_obs2: obs2-to-obs1 error for the same metric.',
                        'eps is a small positive guard to avoid division by zero.',
                    ],
                },
            ],
        },
        {
            'title': '5. Scope Mapping in HTML',
            'paragraphs': [
                'Same statistical family may appear under different scopes; scope is part of metric identity.',
            ],
            'formula_blocks': [
                {
                    'name': 'Scope Disambiguation',
                    'equation': 'Metric Identity = (variable, stat, scope, operator)',
                    'notes': [
                        'Examples: DStd[TS] and DStd[Map] are different metrics by scope.',
                        'Naming avoids same-name ambiguity across different statistical objects.',
                    ],
                },
            ],
        },
    ],
}


__all__ = ['METRIC_FORMULA_PAGE']
