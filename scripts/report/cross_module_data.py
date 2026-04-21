# -*- coding: utf-8 -*-
"""Cross-module interactive payload builders for HTML report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import re


_NAN_STRINGS = {'', 'nan', 'na', 'none', 'null', '--', 'n/a'}


def _norm_text(value: Any) -> str:
    """Return compact string representation."""
    return re.sub(r'\s+', ' ', str(value if value is not None else '')).strip()


def _norm_dataset_name(label: Any) -> str:
    """Normalize dataset labels used in HTML scalar tables."""
    txt = _norm_text(label)
    txt_nobase = re.sub(r'\s*\(baseline\)\s*$', '', txt, flags=re.IGNORECASE).strip()
    low = txt_nobase.lower().replace(' ', '')
    if low == 'obs1':
        return 'obs1'
    if low in {'observation1', 'obs1(baseline)', 'obs1baseline'}:
        return 'obs1'
    if low in {'obs2', 'observation2', 'obs2(baseline)', 'obs2baseline'}:
        return 'obs2'
    return txt_nobase if txt_nobase else txt


def _dataset_type(label: str) -> str:
    """Infer dataset category for filtering."""
    low = _norm_text(label).lower()
    if (
        low.startswith('groupmean[')
        or low.startswith('groupmean(')
        or low.startswith('group mean[')
        or low.startswith('group mean(')
        or low.startswith('gm[')
    ):
        return 'group'
    if low.startswith('obs'):
        return 'obs'
    return 'model'


def _parse_numeric(value: Any) -> float:
    """Parse scalar table cell text into float when possible."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else math.nan

    txt = _norm_text(value).lower()
    if txt in _NAN_STRINGS:
        return math.nan

    # Keep exponent, sign, decimal; drop stars and unit symbols in cells.
    txt = txt.replace('%', '')
    txt = txt.replace('*', '')
    txt = txt.replace(',', '')
    try:
        out = float(txt)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


@dataclass
class _WalkContext:
    """Traversal context for nested dual/region payloads."""

    coverage_mode: str = 'base'
    view_mode: str = 'raw'
    domain_kind: str = 'scalar'
    domain_key: str = 'All'


def _iter_domain_rows(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, List[List[Any]]]]:
    """Yield (domain_kind, domain_key, rows) triplets from one table payload."""
    ptype = str(payload.get('type', '')).strip().lower()
    if ptype == 'seasonal_table':
        seasons = payload.get('seasons', {}) if isinstance(payload.get('seasons'), dict) else {}
        order = [str(s) for s in payload.get('season_order', []) if str(s)]
        if not order:
            order = [str(s) for s in seasons.keys()]
        for season in order:
            rows = seasons.get(season, [])
            if isinstance(rows, list):
                yield ('season', season, rows)
        return

    if ptype == 'period_table':
        periods = payload.get('periods', {}) if isinstance(payload.get('periods'), dict) else {}
        order = [str(p) for p in payload.get('period_order', []) if str(p)]
        if not order:
            order = [str(p) for p in periods.keys()]
        for period in order:
            rows = periods.get(period, [])
            if isinstance(rows, list):
                yield ('period', period, rows)
        return

    if ptype == 'phase_table':
        phases = payload.get('phases', {}) if isinstance(payload.get('phases'), dict) else {}
        order = [str(p) for p in payload.get('phase_order', []) if str(p)]
        if not order:
            order = [str(p) for p in phases.keys()]
        for phase in order:
            rows = phases.get(phase, [])
            if isinstance(rows, list):
                yield ('phase', phase, rows)
        return

    rows = payload.get('rows', [])
    if isinstance(rows, list):
        yield ('scalar', 'All', rows)


def _emit_rows(
    out: List[Dict[str, Any]],
    *,
    hemisphere: str,
    module: str,
    region: str,
    payload: Dict[str, Any],
    ctx: _WalkContext,
) -> None:
    """Append flattened records from one leaf table payload."""
    headers = [str(h) for h in payload.get('headers', [])]
    if len(headers) <= 1:
        return
    units = list(payload.get('units', [''] * len(headers)))
    if len(units) < len(headers):
        units = units + [''] * (len(headers) - len(units))

    for domain_kind, domain_key, rows in _iter_domain_rows(payload):
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) <= 1:
                continue
            ds_name = _norm_dataset_name(row[0])
            if not ds_name:
                continue
            ds_type = _dataset_type(ds_name)
            n_cols = min(len(headers), len(units), len(row))
            for idx in range(1, n_cols):
                metric_name = _norm_text(headers[idx])
                if not metric_name:
                    continue
                value = _parse_numeric(row[idx])
                out.append({
                    'hemisphere': hemisphere,
                    'module': module,
                    'region': region,
                    'coverage_mode': ctx.coverage_mode,
                    'view_mode': ctx.view_mode,
                    'domain_kind': domain_kind,
                    'domain_key': str(domain_key),
                    'metric_name': metric_name,
                    'metric_unit': _norm_text(units[idx]) if idx < len(units) else '',
                    'dataset_name': ds_name,
                    'dataset_type': ds_type,
                    'value': (float(value) if math.isfinite(value) else None),
                })


def _walk_table_payload(
    out: List[Dict[str, Any]],
    *,
    hemisphere: str,
    module: str,
    region: str,
    payload: Dict[str, Any],
    ctx: _WalkContext,
) -> None:
    """Recursively traverse nested dual-table payloads."""
    if not isinstance(payload, dict):
        return
    ptype = str(payload.get('type', '')).strip().lower()
    if ptype == 'dual_table':
        sections = payload.get('sections', [])
        if not isinstance(sections, list):
            return
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sid = str(sec.get('id', '')).strip().lower()
            next_ctx = _WalkContext(
                coverage_mode=ctx.coverage_mode,
                view_mode=ctx.view_mode,
                domain_kind=ctx.domain_kind,
                domain_key=ctx.domain_key,
            )
            if sid in {'base', 'matched'}:
                next_ctx.coverage_mode = sid
            elif sid in {'raw', 'diff'}:
                next_ctx.view_mode = sid
            _walk_table_payload(
                out,
                hemisphere=hemisphere,
                module=module,
                region=region,
                payload=sec,
                ctx=next_ctx,
            )
        return

    _emit_rows(
        out,
        hemisphere=hemisphere,
        module=module,
        region=region,
        payload=payload,
        ctx=ctx,
    )


def _flatten_module_payload(hemisphere: str, module: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one module payload into record list."""
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return out

    ptype = str(payload.get('type', '')).strip().lower()
    if ptype.startswith('region_'):
        region_order = [str(r) for r in payload.get('region_order', []) if str(r)]
        regions = payload.get('regions', {}) if isinstance(payload.get('regions'), dict) else {}
        if not region_order:
            region_order = [str(r) for r in regions.keys()]
        for region in region_order:
            reg_payload = regions.get(region, {})
            _walk_table_payload(
                out,
                hemisphere=hemisphere,
                module=module,
                region=region,
                payload=reg_payload if isinstance(reg_payload, dict) else {},
                ctx=_WalkContext(),
            )
        return out

    _walk_table_payload(
        out,
        hemisphere=hemisphere,
        module=module,
        region='All',
        payload=payload,
        ctx=_WalkContext(),
    )
    return out


def _build_catalog(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build lightweight option catalog for UI bootstrap."""
    hms_map: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        hms = rec['hemisphere']
        mod = rec['module']
        region = rec['region']
        cov = rec['coverage_mode']
        view = rec['view_mode']
        dk = rec['domain_key']
        mn = rec['metric_name']
        node = hms_map.setdefault(hms, {})
        mod_node = node.setdefault(mod, {'regions': set(), 'coverage_modes': set(), 'view_modes': set(), 'domain_keys': set(), 'metrics': set()})
        mod_node['regions'].add(region)
        mod_node['coverage_modes'].add(cov)
        mod_node['view_modes'].add(view)
        mod_node['domain_keys'].add(dk)
        mod_node['metrics'].add(mn)

    out: Dict[str, Any] = {}
    for hms, modules in hms_map.items():
        out[hms] = {}
        for mod, vals in modules.items():
            out[hms][mod] = {
                'regions': sorted(vals['regions']),
                'coverage_modes': sorted(vals['coverage_modes']),
                'view_modes': sorted(vals['view_modes']),
                'domain_keys': sorted(vals['domain_keys']),
                'metrics': sorted(vals['metrics']),
            }
    return out


def build_cross_module_payload(metric_tables: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Build JSON-serializable payload for cross-module interactive explorer."""
    records: List[Dict[str, Any]] = []
    tables = metric_tables or {}
    for hemisphere, modules in tables.items():
        if not isinstance(modules, dict):
            continue
        for module, payload in modules.items():
            if not isinstance(module, str):
                continue
            records.extend(_flatten_module_payload(str(hemisphere), module, payload))

    return {
        'schema_version': 'cross-module-v1',
        'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'records': records,
        'catalog': _build_catalog(records),
    }


__all__ = ['build_cross_module_payload']
