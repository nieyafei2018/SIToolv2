# -*- coding: utf-8 -*-
"""HTML report builders for SIToolv2."""

import glob
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as _xml_escape
import yaml

logger = logging.getLogger(__name__)

_INVALID_XML_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_INVALID_SHEET_CHARS = re.compile(r'[:\\/*?\[\]]')


def _clean_xml_text(value: object) -> str:
    """Return XML-safe UTF-8 text for worksheet inline strings."""
    text = '' if value is None else str(value)
    text = _INVALID_XML_CHARS.sub('', text)
    return _xml_escape(text)


def _excel_col_name(col_idx_1based: int) -> str:
    """Convert 1-based column index to Excel letters (1->A, 27->AA)."""
    n = int(col_idx_1based)
    if n < 1:
        return 'A'
    out = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out.append(chr(65 + rem))
    return ''.join(reversed(out))


def _safe_sheet_name(name: str, used: set) -> str:
    """Create a unique Excel sheet name (<=31 chars, invalid chars removed)."""
    base = _INVALID_SHEET_CHARS.sub('_', str(name or 'Sheet'))
    base = base.strip().strip("'")
    if not base:
        base = 'Sheet'
    base = base[:31]
    candidate = base
    idx = 2
    while candidate in used:
        suffix = f'_{idx}'
        head = base[: max(1, 31 - len(suffix))]
        candidate = f'{head}{suffix}'
        idx += 1
    used.add(candidate)
    return candidate


def _worksheet_xml(rows: List[List[object]]) -> str:
    """Build one worksheet XML using inline strings for all non-empty values."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<sheetData>',
    ]
    for ridx, row in enumerate(rows, start=1):
        cells = []
        for cidx, value in enumerate(row, start=1):
            if value is None:
                continue
            sval = str(value)
            if sval == '':
                continue
            ref = f'{_excel_col_name(cidx)}{ridx}'
            txt = _clean_xml_text(sval)
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{txt}</t></is></c>')
        if cells:
            parts.append(f'<row r="{ridx}">{"".join(cells)}</row>')
    parts.append('</sheetData></worksheet>')
    return ''.join(parts)


def _write_simple_xlsx(xlsx_path: str, sheets: List[dict]) -> None:
    """Write a minimal XLSX workbook without third-party dependencies.

    Parameters
    ----------
    xlsx_path
        Target file path.
    sheets
        List of dicts with keys:
          - "name": sheet title
          - "rows": 2D row-major values
    """
    if not sheets:
        sheets = [{'name': 'Sheet1', 'rows': []}]

    used_names = set()
    sheet_defs = []
    for idx, sheet in enumerate(sheets, start=1):
        sname = _safe_sheet_name(sheet.get('name', f'Sheet{idx}'), used_names)
        srows = sheet.get('rows', [])
        if not isinstance(srows, list):
            srows = []
        sheet_defs.append({'id': idx, 'name': sname, 'rows': srows})

    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)

    workbook_sheets_xml = ''.join(
        f'<sheet name="{_clean_xml_text(s["name"])}" sheetId="{s["id"]}" r:id="rId{s["id"]}"/>'
        for s in sheet_defs
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{workbook_sheets_xml}</sheets>'
        '</workbook>'
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + ''.join(
            f'<Relationship Id="rId{s["id"]}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{s["id"]}.xml"/>'
            for s in sheet_defs
        )
        + '<Relationship Id="rId999" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        + ''.join(
            f'<Override PartName="/xl/worksheets/sheet{s["id"]}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for s in sheet_defs
        )
        + '</Types>'
    )

    package_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        '</Relationships>'
    )

    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>SIToolv2</Application>'
        '<DocSecurity>0</DocSecurity>'
        '<ScaleCrop>false</ScaleCrop>'
        '<HeadingPairs><vt:vector size="2" baseType="variant">'
        '<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>'
        f'<vt:variant><vt:i4>{len(sheet_defs)}</vt:i4></vt:variant>'
        '</vt:vector></HeadingPairs>'
        f'<TitlesOfParts><vt:vector size="{len(sheet_defs)}" baseType="lpstr">'
        + ''.join(f'<vt:lpstr>{_clean_xml_text(s["name"])}</vt:lpstr>' for s in sheet_defs)
        + '</vt:vector></TitlesOfParts>'
        '</Properties>'
    )

    from datetime import datetime, timezone

    ts_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>SIToolv2</dc:creator>'
        '<cp:lastModifiedBy>SIToolv2</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{ts_utc}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{ts_utc}</dcterms:modified>'
        '</cp:coreProperties>'
    )

    # Minimal style sheet to satisfy strict Excel readers.
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(xlsx_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types_xml)
        zf.writestr('_rels/.rels', package_rels_xml)
        zf.writestr('docProps/app.xml', app_xml)
        zf.writestr('docProps/core.xml', core_xml)
        zf.writestr('xl/workbook.xml', workbook_xml)
        zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels_xml)
        zf.writestr('xl/styles.xml', styles_xml)
        for s in sheet_defs:
            zf.writestr(f'xl/worksheets/sheet{s["id"]}.xml', _worksheet_xml(s['rows']))


def generate_html_report(case_name: str, output_dir: str,
                         modules_run: list,
                         metric_tables: dict = None,
                         active_hemisphere: str = None) -> None:
    """Generate an interactive HTML summary report with dual-hemisphere tabs.

    Always renders both Arctic (NH) and Antarctic (SH) top-level tabs.

    Parameters
    ----------
    case_name : str
        Evaluation case name (used in the page title).
    output_dir : str
        Base output directory (e.g. ``cases/highres/output``).
    modules_run : list of str
        Ordered list of module names that were evaluated.
    metric_tables : dict, optional
        Nested mapping ``{hms: {module: table_dict}}`` where each table_dict
        has keys 'headers', 'rows', 'units'.
    active_hemisphere : str, optional
        Hemisphere tab to show on load ('nh' or 'sh').
        Defaults to the first hemisphere that has diagnostic images.
    """
    from datetime import datetime

    report_path = os.path.join(output_dir, 'summary_report.html')
    xlsx_dir = output_dir
    os.makedirs(xlsx_dir, exist_ok=True)
    _xlsx_cache = {}

    from scripts.report.cross_module_data import build_cross_module_payload

    cross_payload = build_cross_module_payload(metric_tables or {})
    cross_records = cross_payload.get('records', []) if isinstance(cross_payload, dict) else []
    cross_payload_inline_json = ''
    cross_json_path = os.path.join(output_dir, 'cross_module_metrics.json')
    try:
        cross_payload_compact = json.dumps(cross_payload, ensure_ascii=False, separators=(',', ':'))
        with open(cross_json_path, 'w', encoding='utf-8') as _fcm:
            _fcm.write(cross_payload_compact)
        cross_payload_inline_json = cross_payload_compact.replace('</', '<\\/')
    except Exception as exc:
        logger.warning("Failed to write cross-module payload JSON (%s).", exc)
        cross_records = []
        cross_payload_inline_json = '{"schema_version":"cross-module-v1","records":[],"catalog":{}}'

    cross_hms_has_data = {'nh': False, 'sh': False}
    for rec in cross_records:
        hms = str(rec.get('hemisphere', '')).lower()
        if hms in cross_hms_has_data:
            cross_hms_has_data[hms] = True

    def _obs_placeholder(value: Any, obs_key: str) -> bool:
        txt = str(value if value is not None else '').strip().lower()
        if not txt:
            return True
        compact = re.sub(r'\s+', '', txt)
        key = str(obs_key).lower()
        placeholders = {key, key.replace('obs', 'observation')}
        return compact in placeholders

    def _infer_obs_name_from_ref_file(ref_file: Any) -> str:
        stem = os.path.splitext(os.path.basename(str(ref_file if ref_file is not None else '')))[0]
        parts = [p for p in stem.split('_') if p]
        if not parts:
            return ''
        stop_tokens = {
            'nh', 'sh', 'arctic', 'antarctic',
            'siconc', 'sidrift', 'sithick', 'sndepth', 'snod',
            'sic', 'sit', 'siu', 'siv', 'sivol', 'sisnthick',
            'daily', 'day', 'mon', 'monthly', 'monmean', 'seasonal',
            'timeseries', 'ts', 'raw', 'diff',
        }
        picked: List[str] = []
        for tok in parts:
            low = tok.lower()
            if re.fullmatch(r'\d{4,8}(?:-\d{4,8})?', low):
                break
            if low in stop_tokens:
                break
            picked.append(tok)
            if len(picked) >= 3:
                break
        if not picked:
            picked = [parts[0]]
        return '_'.join(picked).strip('_')

    def _merge_unique(tokens: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for t in tokens:
            txt = str(t if t is not None else '').strip()
            if not txt:
                continue
            key = txt.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(txt)
        return out

    def _infer_module_obs_labels(module_cfg: Dict[str, Any], hms: str) -> Dict[str, str]:
        ref_lists: List[List[Any]] = []
        for k, v in (module_cfg or {}).items():
            key = str(k).strip().lower()
            if not key.startswith(f'ref_{hms}'):
                continue
            if isinstance(v, list):
                ref_lists.append(v)
        inferred: Dict[str, str] = {}
        for idx, obs_key in enumerate(('obs1', 'obs2')):
            cand: List[str] = []
            for arr in ref_lists:
                if idx >= len(arr):
                    continue
                nm = _infer_obs_name_from_ref_file(arr[idx])
                if nm:
                    cand.append(nm)
            merged = _merge_unique(cand)
            if not merged:
                inferred[obs_key] = 'Obs1' if obs_key == 'obs1' else 'Obs2'
            elif len(merged) == 1:
                inferred[obs_key] = merged[0]
            else:
                inferred[obs_key] = '+'.join(merged)
        return inferred

    obs_name_map: Dict[str, Dict[str, Any]] = {}
    try:
        repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        recipe_file = os.path.join(repo_root, 'cases', f'recipe_{case_name}.yml')
        if os.path.isfile(recipe_file):
            with open(recipe_file, 'r', encoding='utf-8') as _rf:
                recipe_cfg = yaml.safe_load(_rf) or {}
            variables_cfg = recipe_cfg.get('variables') or {}
            if isinstance(variables_cfg, dict):
                for mod_name, mod_cfg in variables_cfg.items():
                    if not isinstance(mod_cfg, dict):
                        continue
                    if str(mod_name).lower() == 'common':
                        continue
                    raw_obs_names = mod_cfg.get('obs_names') if isinstance(mod_cfg.get('obs_names'), dict) else {}
                    mod_labels: Dict[str, Any] = {}
                    base_raw: Dict[str, Any] = {}
                    for key in ('obs1', 'obs2'):
                        base_raw[key] = raw_obs_names.get(key)
                    for hms_key in ('nh', 'sh'):
                        hms_raw = raw_obs_names.get(hms_key) if isinstance(raw_obs_names.get(hms_key), dict) else {}
                        inferred = _infer_module_obs_labels(mod_cfg, hms_key)
                        hms_labels: Dict[str, str] = {}
                        for key in ('obs1', 'obs2'):
                            raw_label = hms_raw.get(key, base_raw.get(key))
                            if _obs_placeholder(raw_label, key):
                                hms_labels[key] = inferred.get(key, 'Obs1' if key == 'obs1' else 'Obs2')
                            else:
                                hms_labels[key] = str(raw_label).strip()
                        mod_labels[hms_key] = hms_labels
                    if mod_labels:
                        obs_name_map[str(mod_name)] = mod_labels
    except Exception as exc:
        logger.warning("Failed to parse module obs_names in recipe_%s.yml (%s).", case_name, exc)
    obs_name_map_inline_json = json.dumps(obs_name_map, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')

    def _canon_name(name: Any) -> str:
        return re.sub(r'[^a-z0-9]+', '', str(name if name is not None else '').lower())

    def _name_candidates(name: Any) -> List[str]:
        base = str(name if name is not None else '').strip()
        if not base:
            return []
        cands = [base]
        cands.extend([p.strip() for p in re.split(r'\s*[+,/]\s*', base) if p.strip()])
        out: List[str] = []
        seen = set()
        for c in cands:
            cc = _canon_name(c)
            if not cc or cc in seen:
                continue
            seen.add(cc)
            out.append(c)
        return out

    _obs_kind_cache: Dict[tuple, Dict[str, str]] = {}

    def _record_obs_kind(module_name: Any, hemisphere: Any, dataset_name: Any) -> str:
        mod_key = str(module_name if module_name is not None else '')
        hms_key = str(hemisphere if hemisphere is not None else '').lower()
        cache_key = (mod_key, hms_key)
        kind_map = _obs_kind_cache.get(cache_key)
        if kind_map is None:
            kind_map = {}
            mod_node = obs_name_map.get(mod_key, {}) if isinstance(obs_name_map, dict) else {}
            hms_node = mod_node.get(hms_key, {}) if isinstance(mod_node, dict) else {}
            for obs_key in ('obs1', 'obs2'):
                raw_label = ''
                if isinstance(hms_node, dict):
                    raw_label = str(hms_node.get(obs_key, '')).strip()
                if not raw_label and isinstance(mod_node, dict):
                    raw_label = str(mod_node.get(obs_key, '')).strip()
                for cand in _name_candidates(raw_label):
                    canon = _canon_name(cand)
                    if canon and canon not in kind_map:
                        kind_map[canon] = obs_key
            _obs_kind_cache[cache_key] = kind_map

        ds_canon = _canon_name(dataset_name)
        if not ds_canon:
            return ''
        return kind_map.get(ds_canon, '')

    # Re-tag concrete observation names (e.g., NSIDC/OSI) in cross-module payload.
    if isinstance(cross_payload, dict) and isinstance(cross_records, list):
        for rec in cross_records:
            if not isinstance(rec, dict):
                continue
            obs_kind = _record_obs_kind(
                rec.get('module'),
                rec.get('hemisphere'),
                rec.get('dataset_name'),
            )
            if obs_kind:
                rec['dataset_type'] = 'obs'
                rec['dataset_obs_kind'] = obs_kind
            else:
                base_type = str(rec.get('dataset_type', '')).lower()
                if base_type == 'group':
                    rec['dataset_type'] = 'group'
                else:
                    rec['dataset_type'] = 'obs' if base_type == 'obs' else 'model'
                rec.pop('dataset_obs_kind', None)
        try:
            cross_payload_compact = json.dumps(cross_payload, ensure_ascii=False, separators=(',', ':'))
            with open(cross_json_path, 'w', encoding='utf-8') as _fcm:
                _fcm.write(cross_payload_compact)
            cross_payload_inline_json = cross_payload_compact.replace('</', '<\\/')
        except Exception as exc:
            logger.warning("Failed to refresh cross-module payload JSON with observation typing (%s).", exc)

    def _slug_token(value: object) -> str:
        text = str(value if value is not None else '').strip()
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text)
        text = text.strip('_.')
        return text or 'x'

    def _sheet_rows(headers: list, units: list, rows: list) -> List[List[object]]:
        out: List[List[object]] = []
        out.append([str(v) for v in (headers or [])])
        out.append([str(v) for v in (units or [''] * len(headers or []))])
        for row in (rows or []):
            out.append([('' if v is None else str(v)) for v in row])
        return out

    def _emit_xlsx(file_tokens: List[object], sheets: List[dict]) -> str:
        # Keep call sites intact but switch to on-demand browser-side export.
        # No workbook files are pre-generated on disk.
        key = '__'.join(_slug_token(t) for t in file_tokens if t is not None and str(t) != '')
        if key in _xlsx_cache:
            return _xlsx_cache[key]
        _xlsx_cache[key] = ''
        return ''

    def _download_controls(single_href: str, all_href: str) -> str:
        all_link = all_href or single_href
        return (
            f'<div class="tbl-downloads">'
            f'<button type="button" class="tbl-dl-btn" data-all="{all_link}" '
            f'onclick="downloadAllTables(this)">Download All Tables</button>'
            f'<div class="tbl-download-hint">'
            f'Tip: Save dialog requires a browser with File System Access API in a secure context '
            f'(https or localhost). Otherwise, files download to the browser default folder.'
            f'</div>'
            f'</div>'
        )

    ALL_HMS = [('nh', 'Arctic (NH)'), ('sh', 'Antarctic (SH)')]

    GEO_SECTOR_MODULE = 'GeographicalSector'
    GEO_SECTOR_LABEL = 'Geographical sector'
    CROSS_MODULE = 'CrossModule'
    CROSS_MODULE_LABEL = 'Cross-Module'

    def _is_sector_map_image(file_name: str) -> bool:
        canon = re.sub(r'[^a-z0-9]+', '', str(file_name).lower())
        return canon in {'seaiceregionmappng', 'geosectormappng'}

    # --- Collect PNG images grouped by hemisphere → module ---
    hms_module_images: dict = {}
    hms_sector_map: dict = {}
    hms_sector_map_source: dict = {}
    for hms, _ in ALL_HMS:
        hms_module_images[hms] = {}
        hms_sector_map[hms] = None
        hms_sector_map_source[hms] = ''
        hms_tables = (metric_tables or {}).get(hms, {})
        for module in modules_run:
            module_dir = os.path.join(output_dir, hms, module)
            rel_imgs = []
            if os.path.isdir(module_dir):
                imgs = sorted(glob.glob(os.path.join(module_dir, '*.png')))
                for img_path in imgs:
                    rel_img = os.path.relpath(img_path, output_dir).replace('\\', '/')
                    if _is_sector_map_image(os.path.basename(img_path)):
                        if hms_sector_map[hms] is None:
                            hms_sector_map[hms] = rel_img
                            hms_sector_map_source[hms] = module
                        continue
                    rel_imgs.append(rel_img)
            if rel_imgs or module in hms_tables:
                hms_module_images[hms][module] = rel_imgs

    hms_nav_modules: dict = {}
    for hms, _ in ALL_HMS:
        nav_modules = list(hms_module_images.get(hms, {}).keys())
        if hms_sector_map.get(hms):
            nav_modules = [GEO_SECTOR_MODULE] + nav_modules
        if cross_hms_has_data.get(hms):
            nav_modules = nav_modules + [CROSS_MODULE]
        hms_nav_modules[hms] = nav_modules

    # Determine which hemisphere tab to show on load: caller's preference,
    # or fall back to the first hemisphere that has diagnostic images.
    if active_hemisphere is None:
        active_hemisphere = next(
            (hms for hms, mods in hms_nav_modules.items() if mods), 'nh'
        )

    def _fmt_col(h: str) -> str:
        """Preserve known acronyms; title-case everything else."""
        _ACRONYMS = {'mke', 'iiee', 'sie', 'sia', 'miz', 'pia', 'mae', 'rmse', 'std'}
        return ' '.join(
            w.upper() if w.lower() in _ACRONYMS else w.capitalize()
            for w in h.replace('_', ' ').split()
        )

    def _build_hms_pane(hms: str, hms_label: str) -> str:
        """Return the full HTML for one hemisphere pane (nav sidebar + module panels)."""
        module_images = hms_module_images.get(hms, {})
        module_order = hms_nav_modules.get(hms, [])
        coverage_base_label = 'Original Coverage'
        coverage_matched_label = 'Obs-Matched Coverage'

        if not module_order:
            return (
                f'  <div class="hms-pane{" active" if hms == active_hemisphere else ""}" id="hms-{hms}">\n'
                f'    <div class="empty-hms"><p>No diagnostic data available for this hemisphere '
                f'(Evaluation disabled in recipe).</p></div>\n'
                f'  </div>\n'
            )

        # Metric tables for this hemisphere from the nested dict
        tables = (metric_tables or {}).get(hms, {})
        default_module = next(
            (mm for mm in module_order if mm != GEO_SECTOR_MODULE),
            (module_order[0] if module_order else ''),
        )
        nav_html = '\n'.join(
            f'        <button class="nav-btn{" active" if (hms == active_hemisphere and m == default_module) else ""}" id="btn-{hms}-{m}" '
            f'onclick="showMod(\'{hms}\',\'{m}\')">{(GEO_SECTOR_LABEL if m == GEO_SECTOR_MODULE else (CROSS_MODULE_LABEL if m == CROSS_MODULE else m))}</button>'
            for m in module_order
        )

        panels_html = ''
        for m in module_order:
            panel_class = 'panel active' if (hms == active_hemisphere and m == default_module) else 'panel'
            if m == GEO_SECTOR_MODULE:
                panel_header = ''
                sector_map_path = hms_sector_map.get(hms)
                sector_source = hms_sector_map_source.get(hms, '')
                if sector_map_path:
                    source_note = f' ({sector_source})' if sector_source else ''
                    panel_content = (
                        f'        <section class="plots-section">\n'
                        f'          <h3 class="section-title">Geographical Sector Map</h3>\n'
                        f'          <p class="geo-note">Sea-ice regional partition overview{source_note}</p>\n'
                        f'          <div class="geo-map-wrap">\n'
                        f'            <figure class="geo-figure plot-figure" data-hms="{hms}" '
                        f'data-mod="{GEO_SECTOR_MODULE}" data-kind="map">'
                        f'<a href="{sector_map_path}" class="img-link" data-src="{sector_map_path}">'
                        f'<img src="{sector_map_path}" loading="lazy" alt="Sea Ice Region Map"></a>'
                        f'<figcaption>Sea ice region map</figcaption>'
                        f'</figure>\n'
                        f'          </div>\n'
                        f'        </section>\n'
                    )
                else:
                    panel_content = (
                        f'        <section class="plots-section">\n'
                        f'          <h3 class="section-title">Geographical Sector Map</h3>\n'
                        f'          <p class="geo-note">No regional map available for this hemisphere.</p>\n'
                        f'        </section>\n'
                    )

                panels_html += (
                    f'        <div class="{panel_class}" id="tab-{hms}-{m}">\n'
                    f'{panel_header}'
                    f'{panel_content}'
                    f'        </div>\n'
                )
                continue

            if m == CROSS_MODULE:
                panel_header = ''
                panel_content = (
                    f'        <section class="metrics-section cross-module-section" data-hms="{hms}">\n'
                    f'          <h3 class="section-title">Cross-Module Explorer</h3>\n'
                    f'          <p class="cross-note">Select two module metrics to explore inter-model relationships. '
                    f'When time-resolution mismatches exist, use the coarser side (season > month) for comparison.</p>\n'
                    f'          <div class="cross-layout">\n'
                    f'            <div class="cross-axis-row">\n'
                    f'              <div class="cross-model-filter">\n'
                    f'                <h4>Models</h4>\n'
                    f'                <div class="cross-model-actions">\n'
                    f'                  <button class="cross-mini-btn" type="button" data-hms="{hms}" data-role="models-all">All</button>\n'
                    f'                  <button class="cross-mini-btn" type="button" data-hms="{hms}" data-role="models-none">None</button>\n'
                    f'                  <span class="cross-model-count" data-hms="{hms}" data-role="model-count"></span>\n'
                    f'                </div>\n'
                    f'                <div class="cross-model-list" data-hms="{hms}" data-role="model-list"></div>\n'
                    f'              </div>\n'
                    f'              <div class="cross-axis">\n'
                    f'                <h4>X-Axis</h4>\n'
                    f'                <label>Module</label><select data-hms="{hms}" data-axis="x" data-role="module"></select>\n'
                    f'                <label>Coverage</label><select data-hms="{hms}" data-axis="x" data-role="coverage"></select>\n'
                    f'                <label>View</label><select data-hms="{hms}" data-axis="x" data-role="view"></select>\n'
                    f'                <label>Region</label><select data-hms="{hms}" data-axis="x" data-role="region"></select>\n'
                    f'                <label>Domain</label><select data-hms="{hms}" data-axis="x" data-role="domain"></select>\n'
                    f'                <label>Metric</label><select data-hms="{hms}" data-axis="x" data-role="metric"></select>\n'
                    f'              </div>\n'
                    f'              <div class="cross-axis">\n'
                    f'                <h4>Y-Axis</h4>\n'
                    f'                <label>Module</label><select data-hms="{hms}" data-axis="y" data-role="module"></select>\n'
                    f'                <label>Coverage</label><select data-hms="{hms}" data-axis="y" data-role="coverage"></select>\n'
                    f'                <label>View</label><select data-hms="{hms}" data-axis="y" data-role="view"></select>\n'
                    f'                <label>Region</label><select data-hms="{hms}" data-axis="y" data-role="region"></select>\n'
                    f'                <label>Domain</label><select data-hms="{hms}" data-axis="y" data-role="domain"></select>\n'
                    f'                <label>Metric</label><select data-hms="{hms}" data-axis="y" data-role="metric"></select>\n'
                    f'              </div>\n'
                    f'              <div class="cross-opts">\n'
                    f'                <h4>Options & Stats</h4>\n'
                    f'                <label>Include Obs</label>\n'
                    f'                <label class="cross-inline-check"><input type="checkbox" data-hms="{hms}" data-role="include-obs1"> <span data-hms="{hms}" data-role="obs1-label">Obs1</span></label>\n'
                    f'                <label class="cross-inline-check"><input type="checkbox" data-hms="{hms}" data-role="include-obs2"> <span data-hms="{hms}" data-role="obs2-label">Obs2</span></label>\n'
                    f'                <label class="cross-inline-check"><input type="checkbox" data-hms="{hms}" data-role="include-groupmean" checked> Group Mean</label>\n'
                    f'                <label>p-threshold</label><input type="number" min="0.001" max="0.5" step="0.001" value="0.05" data-hms="{hms}" data-role="p-thresh">\n'
                    f'                <div class="cross-stat" data-hms="{hms}" data-role="stats">n=0</div>\n'
                    f'                <div class="cross-stat" data-hms="{hms}" data-role="stats-extra"></div>\n'
                    f'              </div>\n'
                    f'              <div class="cross-table-panel">\n'
                    f'                <h4>Paired Values</h4>\n'
                    f'                <div class="cross-table-wrap">\n'
                    f'                  <table class="cross-table" data-hms="{hms}" data-role="point-table">\n'
                    f'                    <thead><tr><th>Dataset</th><th>Type</th><th>X</th><th>Y</th></tr></thead><tbody></tbody>\n'
                    f'                  </table>\n'
                    f'                </div>\n'
                    f'                <button class="tbl-dl-btn" type="button" data-hms="{hms}" data-role="download-csv">Download CSV</button>\n'
                    f'              </div>\n'
                    f'            </div>\n'
                    f'            <div class="cross-main">\n'
                    f'              <div class="cross-plot-wrap">\n'
                    f'                <canvas id="cross-canvas-{hms}" width="1200" height="1200"></canvas>\n'
                    f'                <button class="cross-export-btn" type="button" data-hms="{hms}" data-role="export-png">Export PNG</button>\n'
                    f'              </div>\n'
                    f'            </div>\n'
                    f'          </div>\n'
                    f'        </section>\n'
                )
                panels_html += (
                    f'        <div class="{panel_class}" id="tab-{hms}-{m}">\n'
                    f'{panel_header}'
                    f'{panel_content}'
                    f'        </div>\n'
                )
                continue

            imgs = module_images.get(m, [])
            panel_header = ''

            tbl_section = ''
            tbl_section_base = ''
            tbl_section_matched = ''
            table_type = ''
            table_has_view_base = False
            table_has_view_matched = False
            def _render_table_block(headers: list, units: list, rows: list,
                                    highlight_dadt: bool = False,
                                    single_href: str = '',
                                    all_href: str = '') -> str:
                def _normalize_dataset_label(label: object) -> object:
                    txt = '' if label is None else str(label).strip()
                    norm = re.sub(r'\s+', '', txt).lower()
                    txt_nobase = re.sub(r'\s*\(baseline\)\s*$', '', txt, flags=re.IGNORECASE).strip()

                    def _canon_name(name: object) -> str:
                        return re.sub(r'[^a-z0-9]+', '', str(name if name is not None else '').lower())

                    def _name_candidates(name: str) -> List[str]:
                        base = str(name if name is not None else '').strip()
                        if not base:
                            return []
                        cands = [base]
                        cands.extend([p.strip() for p in re.split(r'\s*[+,/]\s*', base) if p.strip()])
                        out: List[str] = []
                        seen = set()
                        for c in cands:
                            cc = _canon_name(c)
                            if not cc or cc in seen:
                                continue
                            seen.add(cc)
                            out.append(c)
                        return out

                    module_obs = obs_name_map.get(str(m), {}) if isinstance(obs_name_map, dict) else {}
                    hms_obs = module_obs.get(str(hms).lower(), {}) if isinstance(module_obs, dict) else {}
                    obs1_name = str(hms_obs.get('obs1', 'obs1')).strip() if isinstance(hms_obs, dict) else 'obs1'
                    obs2_name = str(hms_obs.get('obs2', 'obs2')).strip() if isinstance(hms_obs, dict) else 'obs2'
                    if not obs1_name:
                        obs1_name = 'obs1'
                    if not obs2_name:
                        obs2_name = 'obs2'

                    obs1_alias = {
                        'obs1', 'observation1',
                        'obs1(baseline)', 'obs1baseline',
                        'observation1(baseline)', 'observation1baseline',
                    }
                    obs2_alias = {
                        'obs2', 'observation2',
                        'obs2(baseline)', 'obs2baseline',
                        'observation2(baseline)', 'observation2baseline',
                    }
                    obs1_cand = _name_candidates(obs1_name)
                    obs2_cand = _name_candidates(obs2_name)
                    obs1_canon = {_canon_name(x) for x in obs1_cand}
                    obs2_canon = {_canon_name(x) for x in obs2_cand}
                    txt_canon = _canon_name(txt_nobase)

                    # Prefer explicit module observation names when present.
                    if txt_canon and txt_canon in obs1_canon:
                        # Keep the concrete row name as baseline if available
                        # (useful when obs1_name is a combined alias like A+B).
                        return f'{txt_nobase} (baseline)'
                    if txt_canon and txt_canon in obs2_canon:
                        # obs2 must never carry the baseline tag.
                        return txt_nobase

                    if norm in obs1_alias:
                        return f'{obs1_name} (baseline)'
                    if norm in obs2_alias:
                        return obs2_name

                    # For any residual "(baseline)" tag on non-obs1 rows, drop it.
                    if txt != txt_nobase:
                        return txt_nobase
                    return label

                header_row = ''.join(
                    f'<th class="sortable" data-col="{idx}">{_fmt_col(h)}</th>'
                    for idx, h in enumerate(headers)
                )
                unit_row = ''.join(
                    f'<th>{u if u else "&nbsp;"}</th>'
                    for u in units
                )
                body_rows = ''
                for row in rows:
                    if isinstance(row, (list, tuple)):
                        row_vals = list(row)
                    else:
                        continue
                    if row_vals:
                        row_vals[0] = _normalize_dataset_label(row_vals[0])
                    cls = ' class="obs-row"' if row_vals and 'baseline' in str(row_vals[0]).lower() else ''
                    row_cells = ''.join(
                        f'<td class="sicb-dadt">{v}</td>' if idx == 1 and highlight_dadt else f'<td>{v}</td>'
                        for idx, v in enumerate(row_vals)
                    )
                    body_rows += f'<tr{cls}>' + row_cells + '</tr>'

                dl_html = _download_controls(single_href, all_href)
                return (
                    f'          <div class="tbl-wrap"><table>\n'
                    f'            <thead>'
                    f'<tr class="col-header">{header_row}</tr>'
                    f'<tr class="unit-row">{unit_row}</tr>'
                    f'</thead>\n'
                    f'            <tbody>{body_rows}</tbody>\n'
                    f'          </table></div>\n'
                    f'          {dl_html}\n'
                )

            def _render_table_markup(table_payload: dict) -> str:
                headers = table_payload.get('headers', [])
                units = table_payload.get('units', [''] * len(headers))
                rows = table_payload.get('rows', [])
                return _render_table_block(
                    headers=headers,
                    units=units,
                    rows=rows,
                    highlight_dadt=bool(table_payload.get('_highlight_dadt')),
                    single_href=str(table_payload.get('_dl_single', '')),
                    all_href=str(table_payload.get('_dl_all', '')),
                )

            def _render_basic_table(table_payload: dict, title: str = 'Scalar Metrics Summary',
                                    download_key: str = 'basic',
                                    sheet_name: str = 'Table',
                                    all_sheets: Optional[List[dict]] = None,
                                    wrap_section: bool = True) -> str:
                headers = list(table_payload.get('headers', []))
                units = list(table_payload.get('units', [''] * len(headers)))
                rows = list(table_payload.get('rows', []))
                one_sheet = {'name': sheet_name, 'rows': _sheet_rows(headers, units, rows)}
                single_href = _emit_xlsx(
                    [hms, m, download_key, 'single'],
                    [one_sheet],
                )
                all_href = _emit_xlsx(
                    [hms, m, download_key, 'all'],
                    all_sheets or [one_sheet],
                )
                payload = dict(table_payload)
                payload['_dl_single'] = single_href
                payload['_dl_all'] = all_href
                body = _render_table_markup(payload)
                if not wrap_section:
                    return body
                return (
                    f'        <section class="metrics-section">\n'
                    f'          <h3 class="section-title">{title}</h3>\n'
                    f'{body}'
                    f'        </section>\n'
                )

            def _normalize_region_payload(region_payload: dict):
                region_tables = region_payload.get('regions', {})
                region_order = [
                    r for r in region_payload.get('region_order', [])
                    if r in region_tables
                ]
                if not region_order:
                    region_order = list(region_tables.keys())
                region_labels = region_payload.get('region_labels', {})
                return region_tables, region_order, region_labels

            def _render_region_tables(region_payload: dict, renderer,
                                      coverage_mode: Optional[str] = None,
                                      view_mode: Optional[str] = None,
                                      wrap_section: bool = True,
                                      title: str = 'Scalar Metrics Summary') -> str:
                region_tables, region_order, region_labels = _normalize_region_payload(region_payload)
                cov_attr = f' data-cov="{coverage_mode}"' if coverage_mode else ''
                view_attr = f' data-view="{view_mode}"' if view_mode else ''
                region_tab_buttons = ''
                for idx, region_key in enumerate(region_order):
                    active_cls = ' active' if idx == 0 else ''
                    region_tab_buttons += (
                        f'<button class="region-tab{active_cls}" '
                        f'data-hms="{hms}" data-mod="{m}" data-region="{region_key}"{cov_attr}{view_attr} '
                        f'onclick="showRegionTable(\'{hms}\',\'{m}\',\'{region_key}\')">'
                        f'{region_labels.get(region_key, region_key)}'
                        f'</button>'
                    )
                region_panes_html = ''
                for idx, region_key in enumerate(region_order):
                    active_cls = ' active' if idx == 0 else ''
                    region_panes_html += (
                        f'<div class="region-pane{active_cls}" data-hms="{hms}" data-mod="{m}" data-region="{region_key}"{cov_attr}{view_attr}>'
                        f'{renderer(region_tables.get(region_key, {}), region_key)}'
                        f'</div>'
                    )
                body = (
                    f'          <div class="region-tabs">{region_tab_buttons}</div>\n'
                    f'          <div class="region-wrap">{region_panes_html}</div>\n'
                )
                if not wrap_section:
                    return body
                return (
                    f'        <section class="metrics-section">\n'
                    f'          <h3 class="section-title">{title}</h3>\n'
                    f'{body}'
                    f'        </section>\n'
                )

            def _render_region_seasonal_markup(payload: dict, region_key: str,
                                               coverage_mode: Optional[str] = None,
                                               view_mode: Optional[str] = None,
                                               all_href_override: Optional[str] = None,
                                               workbook_tag: str = 'seasonal') -> str:
                if payload.get('type') != 'seasonal_table':
                    return _render_table_markup(payload)

                season_order = [
                    str(s) for s in payload.get('season_order', [])
                    if isinstance(s, str) and s
                ] or ['Spring', 'Summer', 'Autumn', 'Winter']
                headers = list(payload.get('headers', []))
                units = list(payload.get('units', [''] * len(headers)))
                cov_attr = f' data-cov="{coverage_mode}"' if coverage_mode else ''
                view_attr = f' data-view="{view_mode}"' if view_mode else ''
                season_tab_buttons = ''
                for idx, season in enumerate(season_order):
                    active_cls = ' active' if idx == 0 else ''
                    season_tab_buttons += (
                        f'<button class="region-season-tab{active_cls}" '
                        f'data-hms="{hms}" data-mod="{m}" data-region="{region_key}" data-season="{season}"{cov_attr}{view_attr} '
                        f'onclick="showRegionSeason(\'{hms}\',\'{m}\',\'{region_key}\',\'{season}\')">{season}</button>'
                    )

                all_sheets = [
                    {'name': season, 'rows': _sheet_rows(headers, units, payload.get('seasons', {}).get(season, []))}
                    for season in season_order
                ]
                all_href = all_href_override or _emit_xlsx(
                    [hms, m, region_key, coverage_mode or 'single', workbook_tag, 'all'],
                    all_sheets,
                )

                season_tables_html = ''
                for idx, season in enumerate(season_order):
                    active_cls = ' active' if idx == 0 else ''
                    season_rows = payload.get('seasons', {}).get(season, [])
                    single_href = _emit_xlsx(
                        [hms, m, region_key, coverage_mode or 'single', view_mode or 'single', workbook_tag, season, 'single'],
                        [{'name': season, 'rows': _sheet_rows(headers, units, season_rows)}],
                    )
                    table_block = _render_table_block(
                        headers=headers,
                        units=units,
                        rows=season_rows,
                        highlight_dadt=True,
                        single_href=single_href,
                        all_href=all_href,
                    )
                    season_tables_html += (
                        f'<div class="region-season-pane{active_cls}" '
                        f'data-hms="{hms}" data-mod="{m}" data-region="{region_key}" data-season="{season}"{cov_attr}{view_attr}>'
                        f'{table_block}'
                        f'</div>'
                    )

                return (
                    f'<div class="region-season-tabs">{season_tab_buttons}</div>'
                    f'<div class="region-season-wrap">{season_tables_html}</div>'
                )

            def _render_region_phase_payload(payload: dict, region_key: str,
                                             coverage_mode: Optional[str] = None,
                                             view_mode: Optional[str] = None) -> str:
                if payload.get('type') != 'phase_table':
                    return _render_table_markup(payload)

                phase_order = ['Advance', 'Retreat']
                headers = list(payload.get('headers', []))
                units = list(payload.get('units', [''] * len(headers)))
                cov_attr = f' data-cov="{coverage_mode}"' if coverage_mode else ''
                view_attr = f' data-view="{view_mode}"' if view_mode else ''
                phase_tab_buttons = ''
                for idx, phase in enumerate(phase_order):
                    active_cls = ' active' if idx == 0 else ''
                    phase_tab_buttons += (
                        f'<button class="region-phase-tab{active_cls}" '
                        f'data-hms="{hms}" data-mod="{m}" data-region="{region_key}" data-phase="{phase}"{cov_attr}{view_attr} '
                        f'onclick="showRegionPhase(\'{hms}\',\'{m}\',\'{region_key}\',\'{phase}\')">{phase}</button>'
                    )

                all_sheets = [
                    {'name': phase, 'rows': _sheet_rows(headers, units, payload.get('phases', {}).get(phase, []))}
                    for phase in phase_order
                ]
                all_href = _emit_xlsx(
                    [hms, m, region_key, coverage_mode or 'single', view_mode or 'single', 'phase', 'all'],
                    all_sheets,
                )

                phase_tables_html = ''
                for idx, phase in enumerate(phase_order):
                    active_cls = ' active' if idx == 0 else ''
                    phase_rows = payload.get('phases', {}).get(phase, [])
                    single_href = _emit_xlsx(
                        [hms, m, region_key, coverage_mode or 'single', view_mode or 'single', 'phase', phase, 'single'],
                        [{'name': phase, 'rows': _sheet_rows(headers, units, phase_rows)}],
                    )
                    table_block = _render_table_block(
                        headers=headers,
                        units=units,
                        rows=phase_rows,
                        highlight_dadt=False,
                        single_href=single_href,
                        all_href=all_href,
                    )
                    phase_tables_html += (
                        f'<div class="region-phase-pane{active_cls}" '
                        f'data-hms="{hms}" data-mod="{m}" data-region="{region_key}" data-phase="{phase}"{cov_attr}{view_attr}>'
                        f'{table_block}'
                        f'</div>'
                    )

                return (
                    f'<div class="region-phase-tabs">{phase_tab_buttons}</div>'
                    f'<div class="region-phase-wrap">{phase_tables_html}</div>'
                )

            def _dual_section_ids(payload: dict) -> List[str]:
                if not isinstance(payload, dict):
                    return []
                sections = payload.get('sections', [])
                out: List[str] = []
                for sec in sections:
                    if not isinstance(sec, dict):
                        continue
                    sid = str(sec.get('id', '')).strip().lower()
                    if sid:
                        out.append(sid)
                return out

            def _dual_kind(payload: dict) -> str:
                ids = set(_dual_section_ids(payload))
                if 'diff' in ids:
                    return 'view'
                if 'matched' in ids or 'base' in ids:
                    return 'coverage'
                return ''

            def _select_dual_section(payload: dict, mode: str) -> Optional[dict]:
                if not isinstance(payload, dict) or payload.get('type') != 'dual_table':
                    return None
                sections = [sec for sec in payload.get('sections', []) if isinstance(sec, dict)]
                if not sections:
                    return None
                selected = next((sec for sec in sections if str(sec.get('id', '')).lower() == str(mode).lower()), None)
                if selected is not None:
                    return selected
                if mode == 'base':
                    selected = next((sec for sec in sections if str(sec.get('id', '')).lower() in ('raw', 'base')), None)
                    if selected is not None:
                        return selected
                if mode == 'matched':
                    selected = next((sec for sec in sections if str(sec.get('id', '')).lower() == 'matched'), None)
                    if selected is not None:
                        return selected
                if mode == 'raw':
                    selected = next((sec for sec in sections if str(sec.get('id', '')).lower() in ('raw', 'base')), None)
                    if selected is not None:
                        return selected
                if mode == 'diff':
                    selected = next((sec for sec in sections if str(sec.get('id', '')).lower() == 'diff'), None)
                    if selected is not None:
                        return selected
                return sections[0]

            def _normalize_diff_rows(rows: list, n_cols: int) -> List[List[object]]:
                norm_rows: List[List[object]] = []
                for row in rows or []:
                    if isinstance(row, (list, tuple)):
                        row_vals = list(row)
                    else:
                        continue
                    norm_rows.append(row_vals)

                col_count = int(n_cols) if int(n_cols) > 0 else 0
                if col_count <= 0:
                    col_count = max((len(r) for r in norm_rows), default=1)
                if norm_rows:
                    first = list(norm_rows[0])
                    if len(first) < col_count:
                        first.extend(['0'] * (col_count - len(first)))
                    label0 = str(first[0]).strip() if first else ''
                    first[0] = label0 or 'Baseline'
                    for jj in range(1, col_count):
                        first[jj] = '0'
                    norm_rows[0] = first
                else:
                    norm_rows = [['Baseline'] + ['0'] * max(col_count - 1, 0)]
                return norm_rows

            def _normalize_diff_payload(payload: dict) -> dict:
                if not isinstance(payload, dict):
                    return {}
                out = dict(payload)
                headers = list(payload.get('headers', []))
                n_cols = len(headers)
                ptype = str(payload.get('type', '')).strip().lower()
                if ptype == 'seasonal_table' or 'seasons' in payload:
                    seasons_src = payload.get('seasons', {}) if isinstance(payload.get('seasons', {}), dict) else {}
                    out['seasons'] = {
                        season_name: _normalize_diff_rows(season_rows, n_cols)
                        for season_name, season_rows in seasons_src.items()
                    }
                elif ptype == 'phase_table' or 'phases' in payload:
                    phases_src = payload.get('phases', {}) if isinstance(payload.get('phases', {}), dict) else {}
                    out['phases'] = {
                        phase_name: _normalize_diff_rows(phase_rows, n_cols)
                        for phase_name, phase_rows in phases_src.items()
                    }
                else:
                    out['rows'] = _normalize_diff_rows(payload.get('rows', []), n_cols)
                return out

            def _render_top_view_tabs(group: str,
                                      coverage_mode: Optional[str] = None,
                                      active: bool = True,
                                      extra_cls: str = '') -> str:
                cov_attr = f' data-cov="{coverage_mode}"' if coverage_mode else ''
                wrap_active_cls = ' active' if active else ''
                return (
                    f'          <div class="view-tabs top-view-tabs{wrap_active_cls}{(" " + extra_cls) if extra_cls else ""}" '
                    f'data-hms="{hms}" data-mod="{m}" data-group="{group}"{cov_attr}>\n'
                    f'            <button class="view-tab active" data-hms="{hms}" data-mod="{m}" data-group="{group}" data-view="raw"{cov_attr} '
                    f'onclick="showView(\'{hms}\',\'{m}\',\'{group}\',\'raw\')">Raw Values</button>\n'
                    f'            <button class="view-tab" data-hms="{hms}" data-mod="{m}" data-group="{group}" data-view="diff"{cov_attr} '
                    f'onclick="showView(\'{hms}\',\'{m}\',\'{group}\',\'diff\')">Differences</button>\n'
                    f'          </div>\n'
                )

            def _render_view_switch_block(raw_html: str, diff_html: str, group: str,
                                          coverage_mode: Optional[str] = None,
                                          title: str = 'Scalar Metrics Summary',
                                          extra_html: str = '') -> str:
                cov_attr = f' data-cov="{coverage_mode}"' if coverage_mode else ''
                return (
                    f'        <section class="metrics-section">\n'
                    f'          <h3 class="section-title">{title}</h3>\n'
                    f'          <div class="view-pane active" data-hms="{hms}" data-mod="{m}" data-group="{group}" data-view="raw"{cov_attr}>\n'
                    f'{raw_html}'
                    f'          </div>\n'
                    f'          <div class="view-pane" data-hms="{hms}" data-mod="{m}" data-group="{group}" data-view="diff"{cov_attr}>\n'
                    f'{diff_html}'
                    f'          </div>\n'
                    f'{extra_html}'
                    f'        </section>\n'
                )

            def _render_region_payload_auto(payload: dict, region_key: str,
                                            coverage_mode: Optional[str] = None,
                                            view_mode: Optional[str] = None) -> str:
                if not isinstance(payload, dict):
                    return ''
                payload_selected = _normalize_diff_payload(payload) if view_mode == 'diff' else payload
                ptype = str(payload_selected.get('type', ''))
                if ptype == 'seasonal_table' or 'seasons' in payload_selected:
                    payload_selected = {
                        'type': 'seasonal_table',
                        'season_order': payload_selected.get('season_order', []),
                        'headers': payload_selected.get('headers', []),
                        'units': payload_selected.get('units', []),
                        'rows': payload_selected.get('rows', []),
                        'seasons': payload_selected.get('seasons', {}),
                    }
                    return _render_region_seasonal_markup(
                        payload_selected,
                        region_key,
                        coverage_mode=coverage_mode,
                        view_mode=view_mode,
                        workbook_tag=f'seasonal_{view_mode or "single"}',
                    )
                if ptype == 'phase_table' or 'phases' in payload_selected:
                    return _render_region_phase_payload(
                        payload_selected,
                        region_key,
                        coverage_mode=coverage_mode,
                        view_mode=view_mode,
                    )

                headers = list(payload_selected.get('headers', []))
                units = list(payload_selected.get('units', [''] * len(headers)))
                rows = list(payload_selected.get('rows', []))
                single_href = _emit_xlsx(
                    [hms, m, region_key, coverage_mode or 'single', view_mode or 'single', 'basic', 'single'],
                    [{'name': 'Table', 'rows': _sheet_rows(headers, units, rows)}],
                )
                all_href = _emit_xlsx(
                    [hms, m, region_key, coverage_mode or 'single', view_mode or 'single', 'basic', 'all'],
                    [{'name': 'Table', 'rows': _sheet_rows(headers, units, rows)}],
                )
                return _render_table_markup({
                    'headers': headers,
                    'units': units,
                    'rows': rows,
                    '_dl_single': single_href,
                    '_dl_all': all_href,
                })

            if tables and m in tables:
                t = tables[m]
                table_type = str(t.get('type', ''))

                if table_type == 'dual_table':
                    kind = _dual_kind(t)
                    if kind == 'view':
                        table_has_view_base = True
                        raw_sec = _select_dual_section(t, 'raw')
                        diff_sec = _normalize_diff_payload(_select_dual_section(t, 'diff') or {})
                        extra_sections = [
                            sec for sec in (t.get('sections', []) or [])
                            if isinstance(sec, dict) and str(sec.get('id', '')).lower() not in {'raw', 'diff'}
                        ]
                        raw_html = _render_basic_table(
                            raw_sec or {},
                            title='Raw Values',
                            download_key='dual_raw',
                            sheet_name='Raw Values',
                            wrap_section=False,
                        )
                        diff_html = _render_basic_table(
                            diff_sec,
                            title='Differences',
                            download_key='dual_diff',
                            sheet_name='Differences',
                            wrap_section=False,
                        )
                        extra_html = ''
                        for idx, sec in enumerate(extra_sections):
                            sec_title = str(sec.get('title') or f'Extended Table {idx + 1}')
                            sec_payload = dict(sec)
                            sec_payload.pop('id', None)
                            sec_payload.pop('title', None)
                            extra_html += _render_basic_table(
                                sec_payload,
                                title=sec_title,
                                download_key=f'dual_extra_{idx + 1}',
                                sheet_name=sec_title,
                            )
                        tbl_section = _render_view_switch_block(
                            raw_html=raw_html,
                            diff_html=diff_html,
                            group='base',
                            coverage_mode=None,
                            title='Scalar Metrics Summary',
                            extra_html=extra_html,
                        )
                    elif kind == 'coverage':
                        def _render_coverage_payload(sec_payload: Optional[dict],
                                                     coverage_mode: str) -> tuple:
                            if not isinstance(sec_payload, dict):
                                return '', False
                            if sec_payload.get('type') == 'dual_table' and _dual_kind(sec_payload) == 'view':
                                raw_inner = _select_dual_section(sec_payload, 'raw')
                                diff_inner = _normalize_diff_payload(_select_dual_section(sec_payload, 'diff') or {})
                                raw_html = _render_basic_table(
                                    raw_inner or {},
                                    title='Raw Values',
                                    download_key=f'dual_{coverage_mode}_raw',
                                    sheet_name='Raw Values',
                                    wrap_section=False,
                                )
                                diff_html = _render_basic_table(
                                    diff_inner,
                                    title='Differences',
                                    download_key=f'dual_{coverage_mode}_diff',
                                    sheet_name='Differences',
                                    wrap_section=False,
                                )
                                return _render_view_switch_block(
                                    raw_html=raw_html,
                                    diff_html=diff_html,
                                    group=coverage_mode,
                                    coverage_mode=coverage_mode,
                                    title='Scalar Metrics Summary',
                                ), True
                            return _render_basic_table(
                                sec_payload,
                                title='Scalar Metrics Summary',
                                download_key=f'dual_{coverage_mode}',
                                sheet_name=(coverage_base_label if coverage_mode == 'base' else coverage_matched_label),
                            ), False

                        base_sec = _select_dual_section(t, 'base')
                        matched_sec = _select_dual_section(t, 'matched')
                        tbl_section_base, table_has_view_base = _render_coverage_payload(base_sec, 'base')
                        if matched_sec is not None:
                            tbl_section_matched, table_has_view_matched = _render_coverage_payload(
                                matched_sec,
                                'matched',
                            )
                        else:
                            tbl_section = tbl_section_base
                    else:
                        tbl_section = _render_basic_table(t)
                    extra_tables = t.get('extra_tables', []) if isinstance(t, dict) else []
                    if isinstance(extra_tables, list):
                        for idx, ext in enumerate(extra_tables):
                            if not isinstance(ext, dict):
                                continue
                            ext_title = str(ext.get('title') or f'Extended Table {idx + 1}')
                            ext_payload = dict(ext)
                            ext_payload.pop('title', None)
                            ext_payload.pop('id', None)
                            rendered_ext = _render_basic_table(
                                ext_payload,
                                title=ext_title,
                                download_key=f'dual_extra_{idx + 1}',
                                sheet_name=ext_title,
                            )
                            ext_cov = str(ext.get('coverage_mode', '')).strip().lower()
                            if not ext_cov:
                                ext_title_l = ext_title.lower()
                                if ('obs-matched' in ext_title_l) or ('matched coverage' in ext_title_l):
                                    ext_cov = 'matched'
                                elif ('original coverage' in ext_title_l) or ('base coverage' in ext_title_l):
                                    ext_cov = 'base'

                            if kind == 'coverage':
                                if ext_cov == 'matched':
                                    if tbl_section_matched:
                                        tbl_section_matched += rendered_ext
                                    elif tbl_section_base:
                                        tbl_section_base += rendered_ext
                                    else:
                                        tbl_section += rendered_ext
                                elif ext_cov == 'base':
                                    if tbl_section_base:
                                        tbl_section_base += rendered_ext
                                    else:
                                        tbl_section += rendered_ext
                                else:
                                    if tbl_section_base:
                                        tbl_section_base += rendered_ext
                                    else:
                                        tbl_section += rendered_ext
                            else:
                                if tbl_section_base:
                                    tbl_section_base += rendered_ext
                                else:
                                    tbl_section += rendered_ext
                elif m == 'SItrans' and table_type == 'phase_table':
                    phase_order = ['Advance', 'Retreat']
                    tab_buttons = ''.join(
                        f'<button class="sitrans-phase-tab{" active" if idx == 0 else ""}" '
                        f'data-hms="{hms}" data-phase="{phase}" '
                        f'onclick="showSitransPhase(\'{hms}\',\'{phase}\')">{phase}</button>'
                        for idx, phase in enumerate(phase_order)
                    )

                    all_sheets = [
                        {'name': phase, 'rows': _sheet_rows(t['headers'], t.get('units', [''] * len(t['headers'])),
                                                            t.get('phases', {}).get(phase, []))}
                        for phase in phase_order
                    ]
                    all_href = _emit_xlsx(
                        [hms, m, 'phase', 'all'],
                        all_sheets,
                    )

                    phase_tables_html = ''
                    for idx, phase in enumerate(phase_order):
                        active_cls = ' active' if idx == 0 else ''
                        phase_rows = t.get('phases', {}).get(phase, [])
                        single_href = _emit_xlsx(
                            [hms, m, 'phase', phase, 'single'],
                            [{'name': phase, 'rows': _sheet_rows(
                                t['headers'],
                                t.get('units', [''] * len(t['headers'])),
                                phase_rows,
                            )}],
                        )
                        table_block = _render_table_block(
                            headers=list(t['headers']),
                            units=list(t.get('units', [''] * len(t['headers']))),
                            rows=phase_rows,
                            highlight_dadt=False,
                            single_href=single_href,
                            all_href=all_href,
                        )
                        phase_tables_html += (
                            f'<div class="sitrans-phase-pane{active_cls}" id="sitrans-phase-{hms}-{phase}" '
                            f'data-hms="{hms}" data-phase="{phase}">'
                            f'{table_block}'
                            f'</div>'
                        )

                    tbl_section = (
                        f'        <section class="metrics-section">\n'
                        f'          <h3 class="section-title">Scalar Metrics Summary</h3>\n'
                        f'          <div class="sitrans-phase-tabs">{tab_buttons}</div>\n'
                        f'          <div class="sitrans-phase-wrap">{phase_tables_html}</div>\n'
                        f'        </section>\n'
                    )
                elif m == 'SIconc' and table_type in ('period_table', 'region_period_table'):
                    period_order = ['Annual', 'March', 'September']

                    if table_type == 'region_period_table':
                        region_tables = t.get('regions', {})
                        region_order = [
                            r for r in t.get('region_order', [])
                            if r in region_tables
                        ]
                        if not region_order:
                            region_order = list(region_tables.keys())
                        region_labels = t.get('region_labels', {})
                    else:
                        region_tables = {'__default__': t}
                        region_order = ['__default__']
                        region_labels = {'__default__': 'All Regions'}

                    region_tab_buttons = ''.join(
                        f'<button class="siconc-region-tab" '
                        f'data-hms="{hms}" data-region="{region_key}" '
                        f'onclick="showSiconcRegion(\'{hms}\',\'{region_key}\')">'
                        f'{region_labels.get(region_key, region_key)}'
                        f'</button>'
                        for region_key in region_order
                    ) if region_order else ''

                    global_period_sheets = []
                    for period in period_order:
                        period_sheet_rows: List[List[object]] = []
                        for region_key in region_order:
                            region_payload = region_tables.get(region_key, {})
                            headers = list(region_payload.get('headers', t.get('headers', [])))
                            units = list(region_payload.get('units', t.get('units', [''] * len(headers))))
                            period_rows = list(region_payload.get('periods', {}).get(period, []))
                            period_sheet_rows.append([f'Region: {region_labels.get(region_key, region_key)}'])
                            period_sheet_rows.extend(_sheet_rows(headers, units, period_rows))
                            period_sheet_rows.append([])
                        global_period_sheets.append({'name': period, 'rows': period_sheet_rows})
                    global_all_href = _emit_xlsx(
                        [hms, m, 'period', 'all_regions', 'all'],
                        global_period_sheets,
                    )

                    region_panes_html = ''
                    for region_key in region_order:
                        region_payload = region_tables.get(region_key, {})
                        headers = region_payload.get('headers', t.get('headers', []))
                        units = region_payload.get('units', t.get('units', [''] * len(headers)))

                        period_tab_buttons = ''.join(
                            f'<button class="siconc-period-tab" '
                            f'data-hms="{hms}" data-region="{region_key}" data-period="{period}" '
                            f'onclick="showSiconcPeriod(\'{hms}\',\'{region_key}\',\'{period}\')">{period}</button>'
                            for period in period_order
                        )

                        period_tables_html = ''
                        for period in period_order:
                            period_rows = region_payload.get('periods', {}).get(period, [])
                            single_href = _emit_xlsx(
                                [hms, m, region_key, 'period', period, 'single'],
                                [{'name': period, 'rows': _sheet_rows(headers, units, period_rows)}],
                            )
                            table_block = _render_table_block(
                                headers=headers,
                                units=units,
                                rows=period_rows,
                                highlight_dadt=False,
                                single_href=single_href,
                                all_href=global_all_href,
                            )
                            period_tables_html += (
                                f'<div class="siconc-period-pane" '
                                f'data-hms="{hms}" data-region="{region_key}" data-period="{period}">'
                                f'{table_block}'
                                f'</div>'
                            )

                        region_panes_html += (
                            f'<div class="siconc-region-pane" '
                            f'data-hms="{hms}" data-region="{region_key}">'
                            f'  <div class="siconc-period-tabs">{period_tab_buttons}</div>'
                            f'  <div class="siconc-period-wrap">{period_tables_html}</div>'
                            f'</div>'
                        )

                    tbl_section = (
                        f'        <section class="metrics-section">\n'
                        f'          <h3 class="section-title">Scalar Metrics Summary</h3>\n'
                        f'          <div class="siconc-region-tabs">{region_tab_buttons}</div>\n'
                        f'          <div class="siconc-region-wrap">{region_panes_html}</div>\n'
                        f'        </section>\n'
                    )
                elif m == 'SICB' and table_type == 'seasonal_table':
                    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
                    tab_buttons = ''.join(
                        f'<button class="sicb-season-tab{" active" if idx == 0 else ""}" '
                        f'data-hms="{hms}" data-season="{season}" '
                        f'onclick="showSicbSeason(\'{hms}\',\'{season}\')">{season}</button>'
                        for idx, season in enumerate(season_order)
                    )

                    all_sheets = [
                        {'name': season, 'rows': _sheet_rows(t['headers'], t.get('units', [''] * len(t['headers'])),
                                                             t.get('seasons', {}).get(season, []))}
                        for season in season_order
                    ]
                    all_href = _emit_xlsx(
                        [hms, m, 'season', 'all'],
                        all_sheets,
                    )

                    season_tables_html = ''
                    for idx, season in enumerate(season_order):
                        active_cls = ' active' if idx == 0 else ''
                        season_rows = t.get('seasons', {}).get(season, [])
                        single_href = _emit_xlsx(
                            [hms, m, 'season', season, 'single'],
                            [{'name': season, 'rows': _sheet_rows(
                                t['headers'],
                                t.get('units', [''] * len(t['headers'])),
                                season_rows,
                            )}],
                        )
                        table_block = _render_table_block(
                            headers=list(t['headers']),
                            units=list(t.get('units', [''] * len(t['headers']))),
                            rows=season_rows,
                            highlight_dadt=True,
                            single_href=single_href,
                            all_href=all_href,
                        )
                        season_tables_html += (
                            f'<div class="sicb-season-pane{active_cls}" id="sicb-season-{hms}-{season}" '
                            f'data-hms="{hms}" data-season="{season}">'
                            f'{table_block}'
                            f'</div>'
                        )

                    tbl_section = (
                        f'        <section class="metrics-section">\n'
                        f'          <h3 class="section-title">Scalar Metrics Summary</h3>\n'
                        f'          <div class="sicb-season-tabs">{tab_buttons}</div>\n'
                        f'          <div class="sicb-season-wrap">{season_tables_html}</div>\n'
                        f'        </section>\n'
                    )
                elif table_type == 'region_basic_table':
                    tbl_section = _render_region_tables(
                        t,
                        renderer=lambda payload, region_key: _render_table_markup({
                            **payload,
                            '_dl_single': _emit_xlsx(
                                [hms, m, region_key, 'basic', 'single'],
                                [{
                                    'name': 'Table',
                                    'rows': _sheet_rows(
                                        list(payload.get('headers', [])),
                                        list(payload.get('units', [])),
                                        list(payload.get('rows', [])),
                                    ),
                                }],
                            ),
                            '_dl_all': _emit_xlsx(
                                [hms, m, region_key, 'basic', 'all'],
                                [{
                                    'name': 'Table',
                                    'rows': _sheet_rows(
                                        list(payload.get('headers', [])),
                                        list(payload.get('units', [])),
                                        list(payload.get('rows', [])),
                                    ),
                                }],
                            ),
                        }),
                    )
                elif table_type == 'region_dual_table':
                    region_tables, region_order, region_labels = _normalize_region_payload(t)
                    sample_kind = ''
                    for payload in region_tables.values():
                        if isinstance(payload, dict) and payload.get('type') == 'dual_table':
                            sample_kind = _dual_kind(payload)
                            if sample_kind:
                                break

                    def _container_from_regions(regions_map: Dict[str, Any]) -> dict:
                        return {
                            'type': 'region_dual_table',
                            'region_order': list(region_order),
                            'region_labels': dict(region_labels),
                            'regions': {rk: regions_map.get(rk, {}) for rk in region_order},
                        }

                    def _region_select_view(view_mode: str) -> dict:
                        selected: Dict[str, Any] = {}
                        for rk in region_order:
                            payload = region_tables.get(rk, {})
                            if isinstance(payload, dict) and payload.get('type') == 'dual_table' and _dual_kind(payload) == 'view':
                                selected[rk] = _select_dual_section(payload, view_mode) or {}
                            else:
                                selected[rk] = payload
                        return _container_from_regions(selected)

                    def _region_select_coverage(cov_mode: str) -> dict:
                        selected: Dict[str, Any] = {}
                        for rk in region_order:
                            payload = region_tables.get(rk, {})
                            if isinstance(payload, dict) and payload.get('type') == 'dual_table' and _dual_kind(payload) == 'coverage':
                                selected[rk] = _select_dual_section(payload, cov_mode) or {}
                            else:
                                selected[rk] = payload
                        return _container_from_regions(selected)

                    def _container_has_view_dual(container: dict) -> bool:
                        ctables, corder, _ = _normalize_region_payload(container)
                        for rk in corder:
                            payload = ctables.get(rk, {})
                            if isinstance(payload, dict) and payload.get('type') == 'dual_table' and _dual_kind(payload) == 'view':
                                return True
                        return False

                    def _render_region_container(container: dict,
                                                 coverage_mode: Optional[str],
                                                 group: str) -> str:
                        if _container_has_view_dual(container):
                            raw_container = _container_from_regions({
                                rk: (
                                    _select_dual_section(container.get('regions', {}).get(rk, {}), 'raw') or {}
                                    if isinstance(container.get('regions', {}).get(rk, {}), dict)
                                    and container.get('regions', {}).get(rk, {}).get('type') == 'dual_table'
                                    and _dual_kind(container.get('regions', {}).get(rk, {})) == 'view'
                                    else container.get('regions', {}).get(rk, {})
                                )
                                for rk in region_order
                            })
                            diff_container = _container_from_regions({
                                rk: (
                                    _select_dual_section(container.get('regions', {}).get(rk, {}), 'diff') or {}
                                    if isinstance(container.get('regions', {}).get(rk, {}), dict)
                                    and container.get('regions', {}).get(rk, {}).get('type') == 'dual_table'
                                    and _dual_kind(container.get('regions', {}).get(rk, {})) == 'view'
                                    else container.get('regions', {}).get(rk, {})
                                )
                                for rk in region_order
                            })
                            raw_html = _render_region_tables(
                                raw_container,
                                renderer=lambda payload, region_key: _render_region_payload_auto(
                                    payload, region_key,
                                    coverage_mode=coverage_mode,
                                    view_mode='raw',
                                ),
                                coverage_mode=coverage_mode,
                                view_mode='raw',
                                wrap_section=False,
                            )
                            diff_html = _render_region_tables(
                                diff_container,
                                renderer=lambda payload, region_key: _render_region_payload_auto(
                                    payload, region_key,
                                    coverage_mode=coverage_mode,
                                    view_mode='diff',
                                ),
                                coverage_mode=coverage_mode,
                                view_mode='diff',
                                wrap_section=False,
                            )
                            return _render_view_switch_block(
                                raw_html=raw_html,
                                diff_html=diff_html,
                                group=group,
                                coverage_mode=coverage_mode,
                                title='Scalar Metrics Summary',
                            )
                        return _render_region_tables(
                            container,
                            renderer=lambda payload, region_key: _render_region_payload_auto(
                                payload, region_key,
                                coverage_mode=coverage_mode,
                                view_mode=None,
                            ),
                            coverage_mode=coverage_mode,
                        )

                    if sample_kind == 'view':
                        table_has_view_base = True
                        tbl_section = _render_region_container(t, coverage_mode=None, group='base')
                    elif sample_kind == 'coverage':
                        base_container = _region_select_coverage('base')
                        matched_container = _region_select_coverage('matched')
                        table_has_view_base = _container_has_view_dual(base_container)
                        table_has_view_matched = _container_has_view_dual(matched_container)
                        tbl_section_base = _render_region_container(base_container, coverage_mode='base', group='base')
                        has_region_matched = any(
                            isinstance(payload, dict)
                            and payload.get('type') == 'dual_table'
                            and _dual_kind(payload) == 'coverage'
                            and _select_dual_section(payload, 'matched') is not None
                            for payload in region_tables.values()
                        )
                        if has_region_matched:
                            tbl_section_matched = _render_region_container(
                                matched_container,
                                coverage_mode='matched',
                                group='matched',
                            )
                        else:
                            tbl_section = tbl_section_base
                    else:
                        tbl_section = _render_region_tables(
                            t,
                            renderer=lambda payload, region_key: _render_region_payload_auto(payload, region_key),
                        )
                    extra_tables = t.get('extra_tables', []) if isinstance(t, dict) else []
                    if isinstance(extra_tables, list):
                        for idx, ext in enumerate(extra_tables):
                            if not isinstance(ext, dict):
                                continue
                            ext_title = str(ext.get('title') or f'Extended Table {idx + 1}')
                            ext_payload = dict(ext)
                            ext_payload.pop('title', None)
                            ext_payload.pop('id', None)
                            rendered_ext = _render_basic_table(
                                ext_payload,
                                title=ext_title,
                                download_key=f'region_dual_extra_{idx + 1}',
                                sheet_name=ext_title,
                            )
                            ext_cov = str(ext.get('coverage_mode', '')).strip().lower()
                            if not ext_cov:
                                ext_title_l = ext_title.lower()
                                if ('obs-matched' in ext_title_l) or ('matched coverage' in ext_title_l):
                                    ext_cov = 'matched'
                                elif ('original coverage' in ext_title_l) or ('base coverage' in ext_title_l):
                                    ext_cov = 'base'

                            if sample_kind == 'coverage':
                                if ext_cov == 'matched':
                                    if tbl_section_matched:
                                        tbl_section_matched += rendered_ext
                                    elif tbl_section_base:
                                        tbl_section_base += rendered_ext
                                    else:
                                        tbl_section += rendered_ext
                                elif ext_cov == 'base':
                                    if tbl_section_base:
                                        tbl_section_base += rendered_ext
                                    else:
                                        tbl_section += rendered_ext
                                else:
                                    if tbl_section_base:
                                        tbl_section_base += rendered_ext
                                    else:
                                        tbl_section += rendered_ext
                            else:
                                if tbl_section_base:
                                    tbl_section_base += rendered_ext
                                else:
                                    tbl_section += rendered_ext
                elif table_type == 'region_seasonal_table':
                    region_tables, region_order, region_labels = _normalize_region_payload(t)
                    season_order: List[str] = []
                    for region_key in region_order:
                        payload = region_tables.get(region_key, {})
                        if not isinstance(payload, dict):
                            continue
                        if payload.get('type') != 'seasonal_table' and 'seasons' not in payload:
                            continue
                        season_order = [
                            str(s) for s in payload.get('season_order', [])
                            if isinstance(s, str) and s
                        ]
                        if not season_order and isinstance(payload.get('seasons'), dict):
                            season_order = [str(k) for k in payload.get('seasons', {}).keys()]
                        if season_order:
                            break
                    if not season_order:
                        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

                    all_sheets = []
                    for season in season_order:
                        sheet_rows: List[List[object]] = []
                        for region_key in region_order:
                            payload = region_tables.get(region_key, {})
                            if not isinstance(payload, dict):
                                continue
                            headers = list(payload.get('headers', []))
                            units = list(payload.get('units', [''] * len(headers)))
                            rows = list(payload.get('seasons', {}).get(season, []))
                            sheet_rows.append([f'Region: {region_labels.get(region_key, region_key)}'])
                            sheet_rows.extend(_sheet_rows(headers, units, rows))
                            sheet_rows.append([])
                        all_sheets.append({'name': season, 'rows': sheet_rows})
                    region_season_all_href = _emit_xlsx(
                        [hms, m, 'seasonal', 'all_regions', 'all'],
                        all_sheets,
                    )

                    tbl_section = _render_region_tables(
                        t,
                        renderer=lambda payload, region_key: _render_region_seasonal_markup(
                            payload,
                            region_key,
                            all_href_override=region_season_all_href,
                        ),
                    )
                elif table_type == 'region_phase_table':
                    tbl_section = _render_region_tables(
                        t,
                        renderer=_render_region_phase_payload,
                    )
                else:
                    tbl_section = _render_basic_table(t)

            def _plot_kind(img_path: str) -> str:
                stem = os.path.splitext(os.path.basename(img_path))[0].lower()
                if 'heat_map' in stem or 'heatmap' in stem:
                    return 'heatmap'
                if 'relationship' in stem or 'scatter' in stem or 'regression' in stem:
                    return 'relationship'
                if 'ranking' in stem or 'rank' in stem or 'skill' in stem or '_bar' in stem:
                    return 'ranking'
                line_tokens = (
                    '_ts',
                    'timeseries',
                    '_ano',
                    'iiee',
                    'ridging_ts',
                )
                map_tokens = (
                    '_map',
                    'spatial',
                    'climatology',
                    'trend',
                    'std',
                    'bias',
                    'advance',
                    'retreat',
                )
                if any(tok in stem for tok in line_tokens):
                    return 'line'
                if any(tok in stem for tok in map_tokens):
                    return 'map'
                return 'other'

            def _module_obs_display_names() -> tuple:
                module_obs = obs_name_map.get(str(m), {}) if isinstance(obs_name_map, dict) else {}
                hms_obs = module_obs.get(str(hms).lower(), {}) if isinstance(module_obs, dict) else {}
                obs1_name = str(hms_obs.get('obs1', 'obs1')).strip() if isinstance(hms_obs, dict) else 'obs1'
                obs2_name = str(hms_obs.get('obs2', 'obs2')).strip() if isinstance(hms_obs, dict) else 'obs2'
                return (obs1_name or 'obs1', obs2_name or 'obs2')

            def _replace_obs_tokens(text: str) -> str:
                out = str(text if text is not None else '')
                obs1_name, obs2_name = _module_obs_display_names()
                token_map = (
                    (r'(?i)(?<![A-Za-z0-9])observation1(?![A-Za-z0-9])', obs1_name),
                    (r'(?i)(?<![A-Za-z0-9])obs1(?![A-Za-z0-9])', obs1_name),
                    (r'(?i)(?<![A-Za-z0-9])observation2(?![A-Za-z0-9])', obs2_name),
                    (r'(?i)(?<![A-Za-z0-9])obs2(?![A-Za-z0-9])', obs2_name),
                )
                for pattern, replacement in token_map:
                    out = re.sub(pattern, replacement, out)
                return out

            def _fig_cards(img_list):
                cards = []
                for p in img_list:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    kind = _plot_kind(p)
                    fig_alt = _replace_obs_tokens(stem)
                    fig_caption = _replace_obs_tokens(stem.replace("_", " "))
                    cards.append(
                        f'          <figure class="plot-figure" data-hms="{hms}" data-mod="{m}" '
                        f'data-kind="{kind}">'
                        f'<a href="{p}" class="img-link" data-src="{p}">'
                        f'<img src="{p}" loading="lazy" alt="{fig_alt}"></a>'
                        f'<figcaption>{fig_caption}</figcaption>'
                        f'</figure>'
                    )
                return '\n'.join(cards)

            def _plot_kind_label(kind: str) -> str:
                return {
                    'line': 'Time Series',
                    'map': 'Spatial Maps',
                    'relationship': 'Relationships',
                    'heatmap': 'Heatmap',
                    'ranking': 'Rankings',
                    'other': 'Other Plots',
                }.get(kind, 'Plots')

            def _bucket_plot_kinds(img_list: List[str]) -> Dict[str, List[str]]:
                buckets: Dict[str, List[str]] = {
                    'line': [],
                    'map': [],
                    'relationship': [],
                    'heatmap': [],
                    'ranking': [],
                    'other': [],
                }
                for p in img_list:
                    buckets[_plot_kind(p)].append(p)
                return buckets

            def _plot_kind_order(*buckets: Dict[str, List[str]]) -> List[str]:
                order = ('line', 'map', 'relationship', 'heatmap', 'ranking', 'other')
                return [kind for kind in order if any(b.get(kind) for b in buckets)]

            def _render_plot_cards(img_list: List[str]) -> str:
                if not img_list:
                    return ''
                cards = _fig_cards(img_list)
                return f'            <div class="fig-grid">\n{cards}\n            </div>\n'

            def _coverage_tabs_markup() -> str:
                return (
                    f'          <div class="coverage-tabs">\n'
                    f'            <button class="coverage-tab active" data-hms="{hms}" data-mod="{m}" data-mode="base" '
                    f'onclick="showCoverage(\'{hms}\',\'{m}\',\'base\')">{coverage_base_label}</button>\n'
                    f'            <button class="coverage-tab" data-hms="{hms}" data-mod="{m}" data-mode="matched" '
                    f'onclick="showCoverage(\'{hms}\',\'{m}\',\'matched\')">{coverage_matched_label}</button>\n'
                    f'          </div>\n'
                )

            def _coverage_controls_markup(base_has_view: bool, matched_has_view: bool) -> str:
                view_tabs_markup = ''
                if base_has_view:
                    view_tabs_markup += _render_top_view_tabs(
                        group='base',
                        coverage_mode='base',
                        active=True,
                        extra_cls='coverage-view-tabs',
                    )
                if matched_has_view:
                    view_tabs_markup += _render_top_view_tabs(
                        group='matched',
                        coverage_mode='matched',
                        active=not base_has_view,
                        extra_cls='coverage-view-tabs',
                    )
                view_center = (
                    f'          <div class="coverage-view-center">\n'
                    f'{view_tabs_markup}'
                    f'          </div>\n'
                )
                return (
                    f'          <div class="coverage-controls">\n'
                    f'{_coverage_tabs_markup()}'
                    f'{view_center}'
                    f'            <div class="coverage-controls-spacer" aria-hidden="true"></div>\n'
                    f'          </div>\n'
                )

            def _coverage_tabs_placeholder_markup() -> str:
                return (
                    f'          <div class="coverage-tabs coverage-tabs-placeholder" aria-hidden="true">\n'
                    f'            <button class="coverage-tab" tabindex="-1" type="button">{coverage_base_label}</button>\n'
                    f'            <button class="coverage-tab" tabindex="-1" type="button">{coverage_matched_label}</button>\n'
                    f'          </div>\n'
                )

            def _view_tabs_placeholder_markup() -> str:
                return (
                    f'          <div class="view-tabs top-view-tabs view-tabs-placeholder" aria-hidden="true">\n'
                    f'            <button class="view-tab" tabindex="-1" type="button">Raw Values</button>\n'
                    f'            <button class="view-tab" tabindex="-1" type="button">Differences</button>\n'
                    f'          </div>\n'
                )

            def _module_view_controls_markup(group: str = 'base',
                                             show_view_tabs: bool = True) -> str:
                view_tabs_markup = (
                    _render_top_view_tabs(group=group, extra_cls='module-view-tabs')
                    if show_view_tabs
                    else _view_tabs_placeholder_markup()
                )
                return (
                    f'        <div class="module-controls">\n'
                    f'{_coverage_tabs_placeholder_markup()}'
                    f'          <div class="module-view-center">\n'
                    f'{view_tabs_markup}'
                    f'          </div>\n'
                    f'          <div class="module-controls-spacer" aria-hidden="true"></div>\n'
                    f'        </div>\n'
                )

            def _build_plot_sections(section_imgs: List[str], group: str) -> tuple:
                if not section_imgs:
                    return '', False

                raw_imgs = [p for p in section_imgs if p.endswith('_raw.png')]
                diff_imgs = [p for p in section_imgs if p.endswith('_diff.png')]
                shared_imgs = [p for p in section_imgs if not p.endswith('_raw.png') and not p.endswith('_diff.png')]

                if raw_imgs or diff_imgs:
                    raw_buckets = _bucket_plot_kinds(raw_imgs + shared_imgs)
                    diff_buckets = _bucket_plot_kinds(diff_imgs + shared_imgs)
                    kind_order = _plot_kind_order(raw_buckets, diff_buckets)

                    section_html = ''
                    for kind in kind_order:
                        raw_cards = _render_plot_cards(raw_buckets.get(kind, []))
                        diff_cards = _render_plot_cards(diff_buckets.get(kind, []))
                        if not raw_cards and not diff_cards:
                            continue
                        section_html += (
                            f'        <section class="plots-section">\n'
                            f'          <h3 class="section-title">{_plot_kind_label(kind)}</h3>\n'
                            f'          <div class="view-pane active" id="view-{hms}-{m}-{group}-{kind}-raw" data-hms="{hms}" data-mod="{m}" data-group="{group}" data-view="raw">\n'
                            f'{raw_cards}'
                            f'          </div>\n'
                            f'          <div class="view-pane" id="view-{hms}-{m}-{group}-{kind}-diff" data-hms="{hms}" data-mod="{m}" data-group="{group}" data-view="diff">\n'
                            f'{diff_cards}'
                            f'          </div>\n'
                            f'        </section>\n'
                        )
                    return section_html, True

                buckets = _bucket_plot_kinds(section_imgs)
                kind_order = _plot_kind_order(buckets)
                section_html = ''
                for kind in kind_order:
                    cards = _render_plot_cards(buckets.get(kind, []))
                    if not cards:
                        continue
                    section_html += (
                        f'        <section class="plots-section">\n'
                        f'          <h3 class="section-title">{_plot_kind_label(kind)}</h3>\n'
                        f'{cards}'
                        f'        </section>\n'
                    )
                return section_html, False

            base_imgs = [
                p for p in imgs
                if '_matched' not in os.path.splitext(os.path.basename(p))[0]
            ]
            matched_imgs = [
                p for p in imgs
                if '_matched' in os.path.splitext(os.path.basename(p))[0]
            ]
            if m == 'SICB':
                # SICB no longer supports Obs-Matched mode.
                matched_imgs = []
                tbl_section_matched = ''

            base_fig_section, base_plot_has_view = _build_plot_sections(
                base_imgs, group='base',
            )
            matched_fig_section, matched_plot_has_view = _build_plot_sections(
                matched_imgs, group='matched',
            )
            base_has_view = bool(table_has_view_base or base_plot_has_view)
            matched_has_view = bool(table_has_view_matched or matched_plot_has_view)

            has_coverage_modes = bool(tbl_section_matched or matched_fig_section)
            if has_coverage_modes:
                base_content = (tbl_section_base or tbl_section) + base_fig_section
                matched_table_section = tbl_section_matched
                if table_type.startswith('region_') and not matched_table_section:
                    matched_table_section = tbl_section
                matched_content = matched_table_section + matched_fig_section
                if base_content and matched_content:
                    top_controls = _coverage_controls_markup(base_has_view, matched_has_view)
                    panel_content = (
                        f'        <section class="coverage-section">\n'
                        f'{top_controls}'
                        f'          <div class="coverage-pane active" id="coverage-{hms}-{m}-base" data-hms="{hms}" data-mod="{m}" data-mode="base">\n'
                        f'{base_content}'
                        f'          </div>\n'
                        f'          <div class="coverage-pane" id="coverage-{hms}-{m}-matched" data-hms="{hms}" data-mod="{m}" data-mode="matched">\n'
                        f'{matched_content}'
                        f'          </div>\n'
                        f'        </section>\n'
                    )
                else:
                    panel_content = base_content or matched_content
            else:
                top_view_controls = _module_view_controls_markup('base', show_view_tabs=base_has_view)
                panel_content = top_view_controls + tbl_section + base_fig_section

            panels_html += (
                f'        <div class="{panel_class}" id="tab-{hms}-{m}">\n'
                f'{panel_header}'
                f'{panel_content}'
                f'        </div>\n'
            )

        return (
            f'  <div class="hms-pane{" active" if hms == active_hemisphere else ""}" id="hms-{hms}">\n'
            f'    <div class="layout">\n'
            f'      <nav>\n{nav_html}\n      </nav>\n'
            f'      <main>\n{panels_html}      </main>\n'
            f'    </div>\n'
            f'  </div>\n'
        )

    panes_html = ''.join(_build_hms_pane(hms, label) for hms, label in ALL_HMS)

    hms_tabs_html = '\n'.join(
        f'  <button class="hms-tab{" active" if hms == active_hemisphere else ""}" data-hms="{hms}" onclick="showHms(\'{hms}\')">{label}</button>'
        for hms, label in ALL_HMS
    )

    # JS snippet to initialise the first visible module for every hemisphere that has content
    init_mods_js = '\n'.join(
        f"showMod('{hms}', '{next((mm for mm in mods if mm != GEO_SECTOR_MODULE), mods[0])}');"
        for hms, mods in hms_nav_modules.items()
        if mods
    )

    download_js_path = os.path.join(output_dir, 'report_download.js')
    download_js = r"""(function () {
  'use strict';

  const _enc = new TextEncoder();
  const _invalidSheetChars = /[:\\/*?\[\]]/g;
  const _invalidXmlChars = /[\x00-\x08\x0B\x0C\x0E-\x1F]/g;

  function _slugToken(value) {
    const text = String(value == null ? '' : value).trim();
    return text
      .replace(/\s+/g, '_')
      .replace(/[^A-Za-z0-9_.-]+/g, '_')
      .replace(/^[_\.]+|[_\.]+$/g, '') || 'x';
  }

  function _xmlEscape(value) {
    return String(value == null ? '' : value)
      .replace(_invalidXmlChars, '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  function _safeSheetName(name, used) {
    let base = String(name || 'Sheet').replace(_invalidSheetChars, '_').trim().replace(/^'+|'+$/g, '');
    if (!base) base = 'Sheet';
    base = base.slice(0, 31);
    let cand = base;
    let idx = 2;
    while (used.has(cand)) {
      const suf = `_${idx}`;
      cand = `${base.slice(0, Math.max(1, 31 - suf.length))}${suf}`;
      idx += 1;
    }
    used.add(cand);
    return cand;
  }

  function _excelColName(colIdx1) {
    let n = Math.max(1, Number(colIdx1) || 1);
    let out = '';
    while (n > 0) {
      const rem = (n - 1) % 26;
      out = String.fromCharCode(65 + rem) + out;
      n = Math.floor((n - 1) / 26);
    }
    return out;
  }

  function _tableToRows(table) {
    if (!table) return [];
    return Array.from(table.querySelectorAll('tr')).map(tr =>
      Array.from(tr.children).map(td => (td.textContent || '').trim())
    );
  }

  function _worksheetXml(rows) {
    const parts = [
      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
      '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>',
    ];
    const data = Array.isArray(rows) ? rows : [];
    for (let r = 0; r < data.length; r++) {
      const row = Array.isArray(data[r]) ? data[r] : [];
      const cells = [];
      for (let c = 0; c < row.length; c++) {
        const v = row[c];
        if (v == null) continue;
        const s = String(v);
        if (!s) continue;
        const ref = `${_excelColName(c + 1)}${r + 1}`;
        cells.push(`<c r="${ref}" t="inlineStr"><is><t>${_xmlEscape(s)}</t></is></c>`);
      }
      if (cells.length) {
        parts.push(`<row r="${r + 1}">${cells.join('')}</row>`);
      }
    }
    parts.push('</sheetData></worksheet>');
    return parts.join('');
  }

  function _u16(v) {
    const b = new Uint8Array(2);
    b[0] = v & 0xff;
    b[1] = (v >>> 8) & 0xff;
    return b;
  }

  function _u32(v) {
    const b = new Uint8Array(4);
    b[0] = v & 0xff;
    b[1] = (v >>> 8) & 0xff;
    b[2] = (v >>> 16) & 0xff;
    b[3] = (v >>> 24) & 0xff;
    return b;
  }

  function _concatBytes(parts) {
    const total = parts.reduce((acc, p) => acc + p.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of parts) {
      out.set(p, off);
      off += p.length;
    }
    return out;
  }

  const _crcTable = (() => {
    const t = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let j = 0; j < 8; j++) {
        c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      }
      t[i] = c >>> 0;
    }
    return t;
  })();

  function _crc32(bytes) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) {
      c = _crcTable[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    }
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  function _dosDateTime(now) {
    const d = now instanceof Date ? now : new Date();
    const y = Math.max(1980, d.getFullYear());
    const date = ((y - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    const time = (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2);
    return { date, time };
  }

  function _zipStore(entries) {
    const dt = _dosDateTime(new Date());
    const localParts = [];
    const centralParts = [];
    let offset = 0;

    for (const entry of entries) {
      const nameBytes = _enc.encode(entry.name);
      const dataBytes = entry.data instanceof Uint8Array ? entry.data : _enc.encode(String(entry.data || ''));
      const crc = _crc32(dataBytes);
      const size = dataBytes.length >>> 0;

      const local = _concatBytes([
        _u32(0x04034b50), _u16(20), _u16(0), _u16(0),
        _u16(dt.time), _u16(dt.date), _u32(crc),
        _u32(size), _u32(size), _u16(nameBytes.length), _u16(0),
        nameBytes, dataBytes,
      ]);
      localParts.push(local);

      const central = _concatBytes([
        _u32(0x02014b50), _u16(20), _u16(20), _u16(0), _u16(0),
        _u16(dt.time), _u16(dt.date), _u32(crc),
        _u32(size), _u32(size), _u16(nameBytes.length), _u16(0), _u16(0),
        _u16(0), _u16(0), _u32(0), _u32(offset), nameBytes,
      ]);
      centralParts.push(central);
      offset += local.length;
    }

    const centralSize = centralParts.reduce((acc, p) => acc + p.length, 0);
    const eocd = _concatBytes([
      _u32(0x06054b50), _u16(0), _u16(0),
      _u16(entries.length), _u16(entries.length),
      _u32(centralSize), _u32(offset), _u16(0),
    ]);
    return _concatBytes([...localParts, ...centralParts, eocd]);
  }

  function _xlsxBlob(sheetsInput) {
    const sheets = Array.isArray(sheetsInput) && sheetsInput.length ? sheetsInput : [{ name: 'Sheet1', rows: [] }];
    const used = new Set();
    const defs = sheets.map((s, i) => ({
      id: i + 1,
      name: _safeSheetName(s && s.name ? s.name : `Sheet${i + 1}`, used),
      rows: Array.isArray(s && s.rows) ? s.rows : [],
    }));

    const workbookSheetsXml = defs
      .map(s => `<sheet name="${_xmlEscape(s.name)}" sheetId="${s.id}" r:id="rId${s.id}"/>`)
      .join('');
    const workbookXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ` +
      `xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">` +
      `<sheets>${workbookSheetsXml}</sheets></workbook>`;

    const workbookRelsXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
      defs.map(s =>
        `<Relationship Id="rId${s.id}" ` +
        `Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" ` +
        `Target="worksheets/sheet${s.id}.xml"/>`
      ).join('') +
      `<Relationship Id="rId999" ` +
      `Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" ` +
      `Target="styles.xml"/>` +
      `</Relationships>`;

    const contentTypesXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">` +
      `<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>` +
      `<Default Extension="xml" ContentType="application/xml"/>` +
      `<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>` +
      `<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>` +
      `<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>` +
      `<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>` +
      defs.map(s =>
        `<Override PartName="/xl/worksheets/sheet${s.id}.xml" ` +
        `ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`
      ).join('') +
      `</Types>`;

    const packageRelsXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
      `<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>` +
      `<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>` +
      `<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>` +
      `</Relationships>`;

    const appXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" ` +
      `xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">` +
      `<Application>SIToolv2</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>` +
      `<HeadingPairs><vt:vector size="2" baseType="variant">` +
      `<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>` +
      `<vt:variant><vt:i4>${defs.length}</vt:i4></vt:variant>` +
      `</vt:vector></HeadingPairs>` +
      `<TitlesOfParts><vt:vector size="${defs.length}" baseType="lpstr">` +
      defs.map(s => `<vt:lpstr>${_xmlEscape(s.name)}</vt:lpstr>`).join('') +
      `</vt:vector></TitlesOfParts></Properties>`;

    const ts = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
    const coreXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" ` +
      `xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" ` +
      `xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">` +
      `<dc:creator>SIToolv2</dc:creator><cp:lastModifiedBy>SIToolv2</cp:lastModifiedBy>` +
      `<dcterms:created xsi:type="dcterms:W3CDTF">${ts}</dcterms:created>` +
      `<dcterms:modified xsi:type="dcterms:W3CDTF">${ts}</dcterms:modified>` +
      `</cp:coreProperties>`;

    const stylesXml =
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
      `<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>` +
      `<fills count="1"><fill><patternFill patternType="none"/></fill></fills>` +
      `<borders count="1"><border/></borders>` +
      `<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>` +
      `<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>` +
      `<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>` +
      `</styleSheet>`;

    const entries = [
      { name: '[Content_Types].xml', data: contentTypesXml },
      { name: '_rels/.rels', data: packageRelsXml },
      { name: 'docProps/app.xml', data: appXml },
      { name: 'docProps/core.xml', data: coreXml },
      { name: 'xl/workbook.xml', data: workbookXml },
      { name: 'xl/_rels/workbook.xml.rels', data: workbookRelsXml },
      { name: 'xl/styles.xml', data: stylesXml },
    ];
    defs.forEach(s => {
      entries.push({ name: `xl/worksheets/sheet${s.id}.xml`, data: _worksheetXml(s.rows) });
    });

    const zipBytes = _zipStore(entries);
    return new Blob([zipBytes], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    });
  }

  function _ensureXlsxName(filename) {
    let name = String(filename || 'table.xlsx').trim();
    if (!name) name = 'table.xlsx';
    if (!/\.xlsx$/i.test(name)) {
      name += '.xlsx';
    }
    return name;
  }

  function _fallbackAnchorDownload(blob, filename) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 1200);
  }

  async function _triggerDownload(blob, filename) {
    const safeName = _ensureXlsxName(filename);
    if (typeof window.showSaveFilePicker === 'function' && window.isSecureContext) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: safeName,
          types: [{
            description: 'Excel Workbook',
            accept: {
              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
            },
          }],
          excludeAcceptAllOption: false,
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
        return;
      } catch (err) {
        // User explicitly canceled save dialog: do not force fallback download.
        if (err && err.name === 'AbortError') return;
      }
    }
    _fallbackAnchorDownload(blob, safeName);
  }

  function _inferPanelInfo(btn) {
    let hms = '';
    let mod = '';
    const d = btn.closest('[data-hms][data-mod]');
    if (d) {
      hms = d.dataset.hms || hms;
      mod = d.dataset.mod || mod;
    }
    const panel = btn.closest('.panel');
    if ((!hms || !mod) && panel && panel.id) {
      const m = panel.id.match(/^tab-([^-]+)-(.+)$/);
      if (m) {
        hms = hms || m[1];
        mod = mod || m[2];
      }
    }
    return { hms, mod };
  }

  function _fileName(btn, suffix) {
    const info = _inferPanelInfo(btn);
    const parts = [info.hms || 'h', info.mod || 'module'];
    const pRegion = btn.closest('[data-region]');
    if (pRegion && pRegion.dataset.region) parts.push(pRegion.dataset.region);
    const pCov = btn.closest('[data-cov], [data-mode]');
    if (pCov) {
      const v = pCov.dataset.cov || pCov.dataset.mode;
      if (v) parts.push(v);
    }
    const pView = btn.closest('[data-view]');
    if (pView && pView.dataset.view) parts.push(pView.dataset.view);
    parts.push(suffix || 'table');
    return `${parts.map(_slugToken).join('__')}.xlsx`;
  }

  function _currentTable(btn) {
    const dl = btn.closest('.tbl-downloads');
    if (!dl) return null;
    const prev = dl.previousElementSibling;
    if (!prev) return null;
    return prev.querySelector('table');
  }

  function _tabLabel(btn) {
    const p = btn.closest('[data-period], [data-season], [data-phase]');
    if (!p) return 'Table';
    return p.dataset.period || p.dataset.season || p.dataset.phase || 'Table';
  }

  function _rowsWithRegionHeader(regionLabel, table) {
    const rows = [];
    rows.push([`Region: ${regionLabel}`]);
    rows.push(..._tableToRows(table));
    rows.push([]);
    return rows;
  }

  function _collectSiconcAll(btn) {
    const periodPane = btn.closest('.siconc-period-pane');
    if (!periodPane) return null;
    const regionWrap = btn.closest('.siconc-region-wrap');
    if (!regionWrap) return null;
    const regionPanes = Array.from(regionWrap.querySelectorAll('.siconc-region-pane'));
    if (!regionPanes.length) return null;
    const first = regionPanes[0];
    let periods = Array.from(first.querySelectorAll('.siconc-period-tab'))
      .map(el => el.dataset.period)
      .filter(Boolean);
    if (!periods.length) {
      periods = Array.from(first.querySelectorAll('.siconc-period-pane'))
        .map(el => el.dataset.period)
        .filter(Boolean);
    }
    periods = Array.from(new Set(periods));
    if (!periods.length) return null;

    const sec = btn.closest('.metrics-section') || btn.closest('.panel') || document;
    const sheets = [];
    for (const period of periods) {
      const rows = [];
      for (const regionPane of regionPanes) {
        const region = regionPane.dataset.region || '';
        const tab = sec.querySelector(`.siconc-region-tab[data-region="${region}"]`);
        const label = tab ? (tab.textContent || '').trim() : (region || 'Region');
        const pp = Array.from(regionPane.querySelectorAll('.siconc-period-pane'))
          .find(x => x.dataset.period === period);
        const table = pp ? pp.querySelector('table') : null;
        if (table) rows.push(..._rowsWithRegionHeader(label, table));
      }
      sheets.push({ name: period, rows });
    }
    return sheets;
  }

  function _collectRegionSeasonAll(btn) {
    const seasonPane = btn.closest('.region-season-pane');
    if (!seasonPane) return null;
    const regionWrap = btn.closest('.region-wrap');
    if (!regionWrap) return null;
    const metrics = btn.closest('.metrics-section') || btn.closest('.panel') || document;
    const cov = seasonPane.dataset.cov || '';
    const view = seasonPane.dataset.view || (seasonPane.closest('[data-view]') ? seasonPane.closest('[data-view]').dataset.view || '' : '');
    const regionPanes = Array.from(regionWrap.querySelectorAll('.region-pane'))
      .filter(p => (!cov || p.dataset.cov === cov) && (!view || p.dataset.view === view));
    if (!regionPanes.length) return null;

    const first = regionPanes[0];
    let seasons = Array.from(first.querySelectorAll('.region-season-tab'))
      .filter(el => (!cov || el.dataset.cov === cov) && (!view || el.dataset.view === view))
      .map(el => el.dataset.season)
      .filter(Boolean);
    if (!seasons.length) {
      seasons = Array.from(first.querySelectorAll('.region-season-pane'))
        .filter(el => (!cov || el.dataset.cov === cov) && (!view || el.dataset.view === view))
        .map(el => el.dataset.season)
        .filter(Boolean);
    }
    seasons = Array.from(new Set(seasons));
    if (!seasons.length) return null;

    const sheets = [];
    for (const season of seasons) {
      const rows = [];
      for (const regionPane of regionPanes) {
        const region = regionPane.dataset.region || '';
        let sel = `.region-tab[data-region="${region}"]`;
        if (cov) sel += `[data-cov="${cov}"]`;
        if (view) sel += `[data-view="${view}"]`;
        const tab = metrics.querySelector(sel);
        const label = tab ? (tab.textContent || '').trim() : (region || 'Region');
        const pp = Array.from(regionPane.querySelectorAll('.region-season-pane'))
          .find(x =>
            x.dataset.season === season &&
            (!cov || x.dataset.cov === cov) &&
            (!view || x.dataset.view === view)
          );
        const table = pp ? pp.querySelector('table') : null;
        if (table) rows.push(..._rowsWithRegionHeader(label, table));
      }
      sheets.push({ name: season, rows });
    }
    return sheets;
  }

  function _collectRegionPhaseAll(btn) {
    const phasePane = btn.closest('.region-phase-pane');
    if (!phasePane) return null;
    const regionWrap = btn.closest('.region-wrap');
    if (!regionWrap) return null;
    const metrics = btn.closest('.metrics-section') || btn.closest('.panel') || document;
    const cov = phasePane.dataset.cov || '';
    const view = phasePane.dataset.view || (phasePane.closest('[data-view]') ? phasePane.closest('[data-view]').dataset.view || '' : '');
    const regionPanes = Array.from(regionWrap.querySelectorAll('.region-pane'))
      .filter(p => (!cov || p.dataset.cov === cov) && (!view || p.dataset.view === view));
    if (!regionPanes.length) return null;

    const first = regionPanes[0];
    let phases = Array.from(first.querySelectorAll('.region-phase-tab'))
      .filter(el => (!cov || el.dataset.cov === cov) && (!view || el.dataset.view === view))
      .map(el => el.dataset.phase)
      .filter(Boolean);
    if (!phases.length) {
      phases = Array.from(first.querySelectorAll('.region-phase-pane'))
        .filter(el => (!cov || el.dataset.cov === cov) && (!view || el.dataset.view === view))
        .map(el => el.dataset.phase)
        .filter(Boolean);
    }
    phases = Array.from(new Set(phases));
    if (!phases.length) return null;

    const sheets = [];
    for (const phase of phases) {
      const rows = [];
      for (const regionPane of regionPanes) {
        const region = regionPane.dataset.region || '';
        let sel = `.region-tab[data-region="${region}"]`;
        if (cov) sel += `[data-cov="${cov}"]`;
        if (view) sel += `[data-view="${view}"]`;
        const tab = metrics.querySelector(sel);
        const label = tab ? (tab.textContent || '').trim() : (region || 'Region');
        const pp = Array.from(regionPane.querySelectorAll('.region-phase-pane'))
          .find(x =>
            x.dataset.phase === phase &&
            (!cov || x.dataset.cov === cov) &&
            (!view || x.dataset.view === view)
          );
        const table = pp ? pp.querySelector('table') : null;
        if (table) rows.push(..._rowsWithRegionHeader(label, table));
      }
      sheets.push({ name: phase, rows });
    }
    return sheets;
  }

  function _collectSicbAll(btn) {
    const seasonPane = btn.closest('.sicb-season-pane');
    if (!seasonPane) return null;
    const wrap = btn.closest('.sicb-season-wrap');
    if (!wrap) return null;
    let seasons = Array.from(wrap.querySelectorAll('.sicb-season-tab'))
      .map(el => el.dataset.season)
      .filter(Boolean);
    if (!seasons.length) {
      seasons = Array.from(wrap.querySelectorAll('.sicb-season-pane'))
        .map(el => el.dataset.season)
        .filter(Boolean);
    }
    seasons = Array.from(new Set(seasons));
    const sheets = [];
    for (const season of seasons) {
      const pane = Array.from(wrap.querySelectorAll('.sicb-season-pane'))
        .find(x => x.dataset.season === season);
      const table = pane ? pane.querySelector('table') : null;
      if (table) sheets.push({ name: season, rows: _tableToRows(table) });
    }
    return sheets.length ? sheets : null;
  }

  function _collectSitransAll(btn) {
    const phasePane = btn.closest('.sitrans-phase-pane');
    if (!phasePane) return null;
    const wrap = btn.closest('.sitrans-phase-wrap');
    if (!wrap) return null;
    let phases = Array.from(wrap.querySelectorAll('.sitrans-phase-tab'))
      .map(el => el.dataset.phase)
      .filter(Boolean);
    if (!phases.length) {
      phases = Array.from(wrap.querySelectorAll('.sitrans-phase-pane'))
        .map(el => el.dataset.phase)
        .filter(Boolean);
    }
    phases = Array.from(new Set(phases));
    const sheets = [];
    for (const phase of phases) {
      const pane = Array.from(wrap.querySelectorAll('.sitrans-phase-pane'))
        .find(x => x.dataset.phase === phase);
      const table = pane ? pane.querySelector('table') : null;
      if (table) sheets.push({ name: phase, rows: _tableToRows(table) });
    }
    return sheets.length ? sheets : null;
  }

  function _collectDefaultAll(btn) {
    const table = _currentTable(btn);
    if (!table) return null;
    return [{ name: _tabLabel(btn), rows: _tableToRows(table) }];
  }

  async function _downloadWorkbook(sheets, fileName) {
    if (!Array.isArray(sheets) || !sheets.length) return;
    const blob = _xlsxBlob(sheets);
    await _triggerDownload(blob, fileName);
  }

  window.downloadCurrentTable = async function (btn) {
    const table = _currentTable(btn);
    if (!table) return;
    const sheetName = _tabLabel(btn);
    await _downloadWorkbook([{ name: sheetName, rows: _tableToRows(table) }], _fileName(btn, `single_${sheetName}`));
  };

  window.downloadAllTables = async function (btn) {
    const sheets =
      _collectSiconcAll(btn) ||
      _collectRegionSeasonAll(btn) ||
      _collectRegionPhaseAll(btn) ||
      _collectSicbAll(btn) ||
      _collectSitransAll(btn) ||
      _collectDefaultAll(btn);
    if (!sheets) return;
    await _downloadWorkbook(sheets, _fileName(btn, 'all_tables'));
  };
})();"""
    with open(download_js_path, 'w', encoding='utf-8') as _fjs:
        _fjs.write(download_js)

    cross_js_path = os.path.join(output_dir, 'report_cross_module.js')
    cross_js = r"""(function () {
  'use strict';

  function _query(root, sel) { return (root || document).querySelector(sel); }
  function _queryAll(root, sel) { return Array.from((root || document).querySelectorAll(sel)); }
  function _uniq(values) { return Array.from(new Set(values)); }
  function _num(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }
  function _valueNum(v) {
    if (v == null || v === '') return NaN;
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }
  function _escHtml(v) {
    return String(v == null ? '' : v)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  const _CSS_DPI = 96;
  const _EXPORT_DPI = 300;
  function _domainValue(rec) { return `${rec.domain_kind || 'scalar'}||${rec.domain_key || 'All'}`; }
  function _splitDomain(v) {
    const txt = String(v || 'scalar||All');
    const idx = txt.indexOf('||');
    if (idx < 0) return { kind: 'scalar', key: txt || 'All' };
    return { kind: txt.slice(0, idx) || 'scalar', key: txt.slice(idx + 2) || 'All' };
  }
  function _setSelectOptions(el, options, current) {
    if (!el) return '';
    const vals = Array.isArray(options) ? options : [];
    let selected = (current == null ? '' : String(current));
    if (!vals.length) {
      el.innerHTML = '<option value="">(none)</option>';
      return '';
    }
    if (!vals.includes(selected)) selected = vals[0];
    el.innerHTML = vals.map(v => `<option value="${String(v).replace(/"/g, '&quot;')}">${String(v)}</option>`).join('');
    el.value = selected;
    return selected;
  }
  function _axisEl(hms, axis, role) {
    return _query(document, `select[data-hms="${hms}"][data-axis="${axis}"][data-role="${role}"]`);
  }
  function _axisValue(hms, axis, role) {
    const el = _axisEl(hms, axis, role);
    return el ? String(el.value || '') : '';
  }
  function _panel(hms) {
    return _query(document, `.cross-module-section[data-hms="${hms}"]`);
  }
  function _records(hms) {
    const panel = _panel(hms);
    return panel && Array.isArray(panel._records) ? panel._records : [];
  }
  let _OBS_NAME_MAP_CACHE = null;
  function _obsNameMap() {
    if (_OBS_NAME_MAP_CACHE !== null) return _OBS_NAME_MAP_CACHE;
    const el = document.getElementById('cross-module-obs-names');
    if (!el) {
      _OBS_NAME_MAP_CACHE = {};
      return _OBS_NAME_MAP_CACHE;
    }
    const txt = String(el.textContent || '').trim();
    if (!txt) {
      _OBS_NAME_MAP_CACHE = {};
      return _OBS_NAME_MAP_CACHE;
    }
    try {
      const parsed = JSON.parse(txt);
      _OBS_NAME_MAP_CACHE = (parsed && typeof parsed === 'object') ? parsed : {};
    } catch (err) {
      _OBS_NAME_MAP_CACHE = {};
    }
    return _OBS_NAME_MAP_CACHE;
  }
  function _moduleObsLabel(moduleName, hms, obsKind) {
    const map = _obsNameMap();
    const key = String(moduleName || '');
    const m = map && typeof map === 'object' ? map[key] : null;
    if (!m || typeof m !== 'object') return '';
    const hmsNode = m[String(hms || '').toLowerCase()];
    let raw = null;
    if (hmsNode && typeof hmsNode === 'object') raw = hmsNode[obsKind];
    if (raw == null) raw = m[obsKind];
    const txt = String(raw == null ? '' : raw).trim();
    return txt;
  }
  function _obsLabelForAxisModules(hms, obsKind) {
    const fallback = obsKind === 'obs1' ? 'Obs1' : 'Obs2';
    const xModule = _axisValue(hms, 'x', 'module');
    const yModule = _axisValue(hms, 'y', 'module');
    const xLabel = _moduleObsLabel(xModule, hms, obsKind);
    const yLabel = _moduleObsLabel(yModule, hms, obsKind);
    if (xModule && yModule && xModule === yModule) return xLabel || yLabel || fallback;
    if (xLabel && yLabel) return xLabel === yLabel ? xLabel : `${xLabel} / ${yLabel}`;
    return xLabel || yLabel || fallback;
  }
  function _obsDisplayName(hms, datasetName) {
    const kind = _obsKind(datasetName);
    if (!kind) return String(datasetName || '');
    return _obsLabelForAxisModules(hms, kind);
  }
  function _updateObsLabels(hms) {
    const obs1Txt = _obsLabelForAxisModules(hms, 'obs1');
    const obs2Txt = _obsLabelForAxisModules(hms, 'obs2');
    const obs1El = _query(document, `span[data-hms="${hms}"][data-role="obs1-label"]`);
    const obs2El = _query(document, `span[data-hms="${hms}"][data-role="obs2-label"]`);
    if (obs1El) obs1El.textContent = obs1Txt;
    if (obs2El) obs2El.textContent = obs2Txt;
  }
  function _normDatasetName(name) {
    return String(name || '').trim().toLowerCase();
  }
  function _obsKind(name) {
    const norm = _normDatasetName(name);
    if (norm.startsWith('obs1')) return 'obs1';
    if (norm.startsWith('obs2')) return 'obs2';
    return '';
  }
  function _canonDatasetName(name) {
    return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
  }
  function _obsNameCandidates(name) {
    const base = String(name || '').trim();
    if (!base) return [];
    const raw = [base, ...base.split(/[+,/]/).map(v => String(v || '').trim()).filter(Boolean)];
    const out = [];
    const seen = new Set();
    raw.forEach(v => {
      const canon = _canonDatasetName(v);
      if (!canon || seen.has(canon)) return;
      seen.add(canon);
      out.push(v);
    });
    return out;
  }
  function _recordObsKind(hms, rec) {
    if (!rec || typeof rec !== 'object') return '';
    const explicit = String(rec.dataset_obs_kind || '').toLowerCase();
    if (explicit === 'obs1' || explicit === 'obs2') return explicit;
    const byToken = _obsKind(rec.dataset_name);
    if (byToken) return byToken;

    const moduleName = String(rec.module || '').trim();
    if (!moduleName) return '';
    const map = _obsNameMap();
    const moduleNode = map && typeof map === 'object' ? map[moduleName] : null;
    if (!moduleNode || typeof moduleNode !== 'object') return '';
    const hmsNode = moduleNode[String(hms || '').toLowerCase()];
    const node = (hmsNode && typeof hmsNode === 'object') ? hmsNode : moduleNode;
    const target = _canonDatasetName(rec.dataset_name);
    if (!target) return '';

    const labels = [
      ['obs1', node.obs1],
      ['obs2', node.obs2],
    ];
    for (const [kind, label] of labels) {
      const cands = _obsNameCandidates(label);
      if (cands.some(v => _canonDatasetName(v) === target)) return kind;
    }
    return '';
  }
  function _isObsRecord(hms, rec) {
    if (_recordObsKind(hms, rec)) return true;
    const type = String(rec && rec.dataset_type ? rec.dataset_type : 'model').toLowerCase();
    return type === 'obs';
  }
  function _recordDatasetType(hms, rec) {
    if (_recordObsKind(hms, rec)) return 'obs';
    const type = String(rec && rec.dataset_type ? rec.dataset_type : 'model').toLowerCase();
    if (type === 'group') return 'group';
    if (type === 'obs') return 'obs';
    return 'model';
  }
  function _modelSelectionState(hms) {
    const inputs = _queryAll(document, `input[data-hms="${hms}"][data-role="model-item"]`);
    const selected = new Set(
      inputs.filter(el => !!el.checked).map(el => String(el.value || ''))
    );
    return { selected, total: inputs.length };
  }
  function _updateModelCount(hms) {
    const statEl = _query(document, `span[data-hms="${hms}"][data-role="model-count"]`);
    if (!statEl) return;
    const st = _modelSelectionState(hms);
    statEl.textContent = `${st.selected.size}/${st.total}`;
  }
  function _renderModelSelector(hms) {
    const recs = _records(hms);
    const listEl = _query(document, `div[data-hms="${hms}"][data-role="model-list"]`);
    if (!listEl) return;
    const existingInputs = _queryAll(listEl, `input[data-hms="${hms}"][data-role="model-item"]`);
    const hadExisting = existingInputs.length > 0;
    const prevSelected = new Set(
      existingInputs.filter(el => !!el.checked).map(el => String(el.value || ''))
    );
    const models = _uniq(
      recs.filter(r => {
        const name = String(r.dataset_name || '');
        return !!name && _recordDatasetType(hms, r) === 'model';
      }).map(r => String(r.dataset_name || ''))
    ).sort((a, b) => (a < b ? -1 : (a > b ? 1 : 0)));
    if (!models.length) {
      listEl.innerHTML = '<div class="cross-model-empty">(no models)</div>';
      _updateModelCount(hms);
      return;
    }
    listEl.innerHTML = models.map((name, idx) => {
      const checked = hadExisting ? prevSelected.has(name) : true;
      const checkedAttr = checked ? ' checked' : '';
      const safeId = `cross-model-${hms}-${idx}`;
      const escVal = String(name).replace(/"/g, '&quot;');
      const escTxt = String(name).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return `<label class="cross-model-item" for="${safeId}"><input id="${safeId}" type="checkbox" data-hms="${hms}" data-role="model-item" value="${escVal}"${checkedAttr}><span>${escTxt}</span></label>`;
    }).join('');
    _queryAll(listEl, `input[data-hms="${hms}"][data-role="model-item"]`).forEach(el => {
      el.addEventListener('change', () => {
        _updateModelCount(hms);
        _render(hms);
      });
    });
    _updateModelCount(hms);
  }
  function _setAllModels(hms, checked) {
    _queryAll(document, `input[data-hms="${hms}"][data-role="model-item"]`).forEach(el => {
      el.checked = !!checked;
    });
    _updateModelCount(hms);
    _render(hms);
  }
  function _filteredRecords(hms, axis, includeMetric) {
    const recs = _records(hms);
    const moduleName = _axisValue(hms, axis, 'module');
    const coverage = _axisValue(hms, axis, 'coverage');
    const view = _axisValue(hms, axis, 'view');
    const region = _axisValue(hms, axis, 'region');
    const dom = _splitDomain(_axisValue(hms, axis, 'domain'));
    const metric = _axisValue(hms, axis, 'metric');
    return recs.filter(r =>
      String(r.module) === moduleName &&
      String(r.coverage_mode) === coverage &&
      String(r.view_mode) === view &&
      String(r.region) === region &&
      String(r.domain_kind) === dom.kind &&
      String(r.domain_key) === dom.key &&
      (!includeMetric || String(r.metric_name) === metric)
    );
  }
  function _refreshAxis(hms, axis) {
    const recs = _records(hms);
    if (!recs.length) return;
    const moduleEl = _axisEl(hms, axis, 'module');
    const covEl = _axisEl(hms, axis, 'coverage');
    const viewEl = _axisEl(hms, axis, 'view');
    const regionEl = _axisEl(hms, axis, 'region');
    const domainEl = _axisEl(hms, axis, 'domain');
    const metricEl = _axisEl(hms, axis, 'metric');

    const modules = _uniq(recs.map(r => String(r.module))).sort();
    const moduleSel = _setSelectOptions(moduleEl, modules, moduleEl ? moduleEl.value : '');
    const byMod = recs.filter(r => String(r.module) === moduleSel);

    const covs = _uniq(byMod.map(r => String(r.coverage_mode))).sort();
    const covSel = _setSelectOptions(covEl, covs, covEl ? covEl.value : '');
    const byCov = byMod.filter(r => String(r.coverage_mode) === covSel);

    const views = _uniq(byCov.map(r => String(r.view_mode))).sort();
    const viewSel = _setSelectOptions(viewEl, views, viewEl ? viewEl.value : '');
    const byView = byCov.filter(r => String(r.view_mode) === viewSel);

    const regions = _uniq(byView.map(r => String(r.region))).sort();
    const regionSel = _setSelectOptions(regionEl, regions, regionEl ? regionEl.value : '');
    const byRegion = byView.filter(r => String(r.region) === regionSel);

    const domains = _uniq(byRegion.map(r => _domainValue(r))).sort((a, b) => {
      const aa = _splitDomain(a), bb = _splitDomain(b);
      const ka = `${aa.kind}:${aa.key}`;
      const kb = `${bb.kind}:${bb.key}`;
      return ka < kb ? -1 : (ka > kb ? 1 : 0);
    });
    const domSel = _setSelectOptions(domainEl, domains, domainEl ? domainEl.value : '');
    const domObj = _splitDomain(domSel);
    if (domainEl) {
      Array.from(domainEl.options).forEach(opt => {
        const d = _splitDomain(opt.value);
        opt.textContent = `${d.kind}:${d.key}`;
      });
    }
    const byDomain = byRegion.filter(r =>
      String(r.domain_kind) === domObj.kind && String(r.domain_key) === domObj.key
    );

    const metrics = _uniq(byDomain.map(r => String(r.metric_name))).sort();
    _setSelectOptions(metricEl, metrics, metricEl ? metricEl.value : '');
  }

  function _pearson(xs, ys) {
    const n = xs.length;
    if (n < 2) return NaN;
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
    const mx = sx / n, my = sy / n;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
      const a = xs[i] - mx;
      const b = ys[i] - my;
      num += a * b;
      dx += a * a;
      dy += b * b;
    }
    if (!(dx > 0) || !(dy > 0)) return NaN;
    return num / Math.sqrt(dx * dy);
  }
  function _ranks(values) {
    const n = values.length;
    const pairs = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Array(n).fill(NaN);
    let k = 0;
    while (k < n) {
      let j = k + 1;
      while (j < n && pairs[j].v === pairs[k].v) j++;
      const avg = 0.5 * (k + j - 1) + 1;
      for (let t = k; t < j; t++) ranks[pairs[t].i] = avg;
      k = j;
    }
    return ranks;
  }
  function _spearman(xs, ys) {
    if (xs.length < 2) return NaN;
    return _pearson(_ranks(xs), _ranks(ys));
  }
  function _regression(xs, ys) {
    const n = xs.length;
    if (n < 2) return { slope: NaN, intercept: NaN };
    const mx = xs.reduce((a, b) => a + b, 0) / n;
    const my = ys.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - mx;
      num += dx * (ys[i] - my);
      den += dx * dx;
    }
    if (!(den > 0)) return { slope: NaN, intercept: NaN };
    const slope = num / den;
    return { slope, intercept: my - slope * mx };
  }
  function _tickValues(vmin, vmax, targetCount) {
    const lo = Number(vmin), hi = Number(vmax);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) return [];
    const n = Math.max(3, Number(targetCount) || 6);
    const span = hi - lo;
    const raw = span / (n - 1);
    const p10 = Math.pow(10, Math.floor(Math.log10(Math.abs(raw))));
    const r = raw / p10;
    let step = 1;
    if (r >= 7.5) step = 10;
    else if (r >= 3.5) step = 5;
    else if (r >= 1.5) step = 2;
    step *= p10;
    const start = Math.floor(lo / step) * step;
    const end = Math.ceil(hi / step) * step;
    const out = [];
    for (let v = start; v <= end + step * 0.5; v += step) out.push(v);
    const eps = Math.abs(step) * 1e-9 + 1e-12;
    const inside = out.filter(v => v >= lo - eps && v <= hi + eps);
    if (inside.length > 10) {
      const stride = Math.ceil(inside.length / 10);
      return inside.filter((_, idx) => idx % stride === 0);
    }
    return inside;
  }
  function _fmtTick(v) {
    let x = Number(v);
    if (!Number.isFinite(x)) return '';
    if (Math.abs(x) < 1e-12) x = 0;
    const ax = Math.abs(x);
    let out = '';
    if (ax >= 1000 || (ax > 0 && ax < 0.01)) out = x.toExponential(1);
    else if (ax >= 100) out = x.toFixed(0);
    else if (ax >= 10) out = x.toFixed(1);
    else if (ax >= 1) out = x.toFixed(2);
    else out = x.toFixed(3);
    if (!/[eE]/.test(out)) {
      out = out.replace(/(\.\d*?[1-9])0+$/,'$1').replace(/\.0+$/,'');
      if (out === '-0') out = '0';
    }
    return out;
  }
  function _fmtSig(v, digits) {
    const x = Number(v);
    if (!Number.isFinite(x)) return 'nan';
    const sig = Math.max(1, Number(digits) || 3);
    return Number(x).toPrecision(sig);
  }
  function _fmtCompact(v, decimals) {
    const x = Number(v);
    if (!Number.isFinite(x)) return 'nan';
    const n = Math.max(0, Number(decimals) || 3);
    const ax = Math.abs(x);
    if (ax > 0 && (ax >= 1e4 || ax < Math.pow(10, -n))) {
      const raw = x.toExponential(Math.max(0, n - 1));
      const parts = raw.split('e');
      let mant = parts[0];
      let exp = parts[1] || '';
      mant = mant.replace(/(\.\d*?[1-9])0+$/,'$1').replace(/\.0+$/,'');
      exp = exp.replace(/^\+/, '');
      exp = exp.replace(/^(-?)0+(\d)/, '$1$2');
      const expNum = Number(exp);
      const expTxt = Number.isFinite(expNum) ? String(expNum) : exp;
      return `${mant}e${expTxt}`;
    }
    let out = x.toFixed(n);
    out = out.replace(/(\.\d*?[1-9])0+$/,'$1').replace(/\.0+$/,'');
    if (out === '-0') out = '0';
    return out;
  }
  function _drawStar(ctx, cx, cy, outerRadius, innerRadius, spikes) {
    const n = Math.max(5, Number(spikes) || 5);
    let rot = -Math.PI / 2;
    const step = Math.PI / n;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const ox = cx + Math.cos(rot) * outerRadius;
      const oy = cy + Math.sin(rot) * outerRadius;
      if (i === 0) ctx.moveTo(ox, oy);
      else ctx.lineTo(ox, oy);
      rot += step;
      const ix = cx + Math.cos(rot) * innerRadius;
      const iy = cy + Math.sin(rot) * innerRadius;
      ctx.lineTo(ix, iy);
      rot += step;
    }
    ctx.closePath();
  }
  function _makeRng(seed) {
    let s = (Number(seed) >>> 0) || 1;
    return function () {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  function _seriesSeed(xs, ys) {
    let h = 2166136261 >>> 0;
    const mixToken = (token) => {
      const txt = String(token);
      for (let i = 0; i < txt.length; i++) {
        h ^= txt.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
      }
      h ^= 124;
      h = Math.imul(h, 16777619) >>> 0;
    };
    mixToken(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const x = Number(xs[i]);
      const y = Number(ys[i]);
      mixToken(Number.isFinite(x) ? x.toPrecision(12) : 'nan');
      mixToken(Number.isFinite(y) ? y.toPrecision(12) : 'nan');
    }
    return h || 1;
  }
  function _shuffleInPlace(arr, randFn) {
    const rnd = typeof randFn === 'function' ? randFn : Math.random;
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  function _permPValue(xs, ys, nperm) {
    if (xs.length < 3) return NaN;
    const n = Math.max(200, Number(nperm) || 1200);
    const rObs = Math.abs(_pearson(xs, ys));
    if (!Number.isFinite(rObs)) return NaN;
    const rng = _makeRng(_seriesSeed(xs, ys));
    let cnt = 0;
    for (let i = 0; i < n; i++) {
      const yperm = _shuffleInPlace(ys.slice(), rng);
      const rp = Math.abs(_pearson(xs, yperm));
      if (Number.isFinite(rp) && rp >= rObs) cnt += 1;
    }
    return (cnt + 1) / (n + 1);
  }
  function _axisLabel(hms, axis) {
    const moduleName = _axisValue(hms, axis, 'module');
    const metric = _axisValue(hms, axis, 'metric');
    const dom = _splitDomain(_axisValue(hms, axis, 'domain'));
    const region = _axisValue(hms, axis, 'region');
    const view = _axisValue(hms, axis, 'view');
    const cov = _axisValue(hms, axis, 'coverage');
    return `${moduleName} | ${metric} | ${dom.key} | ${region} | ${cov}/${view}`;
  }
  function _renderPointsTable(hms, points) {
    const body = _query(document, `table.cross-table[data-hms="${hms}"] tbody`);
    if (!body) return;
    body.innerHTML = points.map(p =>
      `<tr><td>${_escHtml(_obsDisplayName(hms, p.dataset))}</td><td>${_escHtml(p.obsKind || p.type)}</td><td>${_fmtCompact(p.x, 3)}</td><td>${_fmtCompact(p.y, 3)}</td></tr>`
    ).join('');
  }
  function _drawScatter(hms, points, xLabel, yLabel, opts) {
    const options = opts || {};
    const canvas = options.canvas || document.getElementById(`cross-canvas-${hms}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dprOpt = Number(options.dpr);
    const dpr = (Number.isFinite(dprOpt) && dprOpt > 0)
      ? dprOpt
      : Math.max(1, Number(window.devicePixelRatio) || 1);
    const rect = (typeof canvas.getBoundingClientRect === 'function')
      ? canvas.getBoundingClientRect()
      : { width: 0, height: 0 };
    const cssW = Math.max(480, Math.round(Number(options.cssW) || rect.width || canvas.clientWidth || 860));
    const cssH = Math.max(480, Math.round(Number(options.cssH) || rect.height || canvas.clientHeight || cssW));
    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cssW, cssH);

    const pad = { l: 138, r: 116, t: 64, b: 104 };
    const pw = cssW - pad.l - pad.r, ph = cssH - pad.t - pad.b;
    if (!options.skipButtonSync) {
      const exportBtn = _query(document, `button[data-hms="${hms}"][data-role="export-png"]`);
      const wrap = (typeof canvas.closest === 'function') ? canvas.closest('.cross-plot-wrap') : null;
      if (exportBtn && wrap) {
        const computed = window.getComputedStyle(wrap);
        const wrapPadL = Number.parseFloat(computed.paddingLeft || '0') || 0;
        const wrapPadT = Number.parseFloat(computed.paddingTop || '0') || 0;
        const btnW = Math.max(82, exportBtn.offsetWidth || 96);
        const btnH = Math.max(28, exportBtn.offsetHeight || 32);
        let left = wrapPadL + pad.l + pw - btnW;
        let top = wrapPadT + pad.t + ph + 60;
        const maxLeft = wrap.clientWidth - btnW - 8;
        const maxTop = wrap.clientHeight - btnH - 8;
        left = Math.max(8, Math.min(left, maxLeft));
        top = Math.max(8, Math.min(top, maxTop));
        exportBtn.style.left = `${Math.round(left)}px`;
        exportBtn.style.top = `${Math.round(top)}px`;
        exportBtn.style.right = 'auto';
        exportBtn.style.bottom = 'auto';
      }
    }
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#8fa2b5';
    ctx.strokeRect(pad.l, pad.t, pw, ph);

    if (!points.length) {
      ctx.fillStyle = '#6b7f92';
      ctx.font = '600 20px sans-serif';
      ctx.fillText('No paired points for current selection.', pad.l + 12, pad.t + 30);
      return;
    }
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    let xmin = Math.min(...xs), xmax = Math.max(...xs);
    let ymin = Math.min(...ys), ymax = Math.max(...ys);
    if (xmin === xmax) { xmin -= 1; xmax += 1; }
    if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const dx = xmax - xmin, dy = ymax - ymin;
    xmin -= dx * 0.14; xmax += dx * 0.14;
    ymin -= dy * 0.12; ymax += dy * 0.12;

    const tx = v => pad.l + (v - xmin) / (xmax - xmin) * pw;
    const ty = v => pad.t + ph - (v - ymin) / (ymax - ymin) * ph;

    const xTicks = _tickValues(xmin, xmax, 6);
    const yTicks = _tickValues(ymin, ymax, 6);
    ctx.lineWidth = 1.1;
    ctx.strokeStyle = '#d4dee9';
    xTicks.forEach(v => {
      const xx = tx(v);
      if (xx < pad.l - 0.5 || xx > pad.l + pw + 0.5) return;
      ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, pad.t + ph); ctx.stroke();
    });
    yTicks.forEach(v => {
      const yy = ty(v);
      if (yy < pad.t - 0.5 || yy > pad.t + ph + 0.5) return;
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
    });

    if (xmin < 0 && xmax > 0) {
      ctx.strokeStyle = '#95a9bd';
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(tx(0), pad.t); ctx.lineTo(tx(0), pad.t + ph); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (ymin < 0 && ymax > 0) {
      ctx.strokeStyle = '#95a9bd';
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, ty(0)); ctx.lineTo(pad.l + pw, ty(0)); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = '#6f869d';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + ph);
    ctx.lineTo(pad.l + pw, pad.t + ph);
    ctx.stroke();

    ctx.fillStyle = '#41586d';
    ctx.font = '600 16px sans-serif';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    xTicks.forEach(v => {
      const xx = tx(v);
      if (xx < pad.l - 0.5 || xx > pad.l + pw + 0.5) return;
      ctx.lineWidth = 1.3;
      ctx.strokeStyle = '#6f869d';
      ctx.beginPath(); ctx.moveTo(xx, pad.t + ph); ctx.lineTo(xx, pad.t + ph + 5); ctx.stroke();
      ctx.fillText(_fmtTick(v), xx, pad.t + ph + 5);
    });
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    yTicks.forEach(v => {
      const yy = ty(v);
      if (yy < pad.t - 0.5 || yy > pad.t + ph + 0.5) return;
      ctx.lineWidth = 1.3;
      ctx.strokeStyle = '#6f869d';
      ctx.beginPath(); ctx.moveTo(pad.l - 5, yy); ctx.lineTo(pad.l, yy); ctx.stroke();
      ctx.fillText(_fmtTick(v), pad.l - 6, yy);
    });

    const reg = _regression(xs, ys);
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, pw, ph);
    ctx.clip();
    if (Number.isFinite(reg.slope) && Number.isFinite(reg.intercept)) {
      const y1 = reg.slope * xmin + reg.intercept;
      const y2 = reg.slope * xmax + reg.intercept;
      ctx.strokeStyle = '#e74c3c';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(tx(xmin), ty(y1)); ctx.lineTo(tx(xmax), ty(y2)); ctx.stroke();
    }

    points.forEach(p => {
      const px = tx(p.x), py = ty(p.y);
      const obsKind = String(p.obsKind || _obsKind(p.dataset)).toLowerCase();
      const isObs1 = obsKind === 'obs1';
      const isObs2 = obsKind === 'obs2';
      if (p.type === 'obs' && (isObs1 || isObs2)) {
        ctx.fillStyle = isObs1 ? '#d62728' : '#4a4f56';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.1;
        _drawStar(ctx, px, py, 8.2, 3.6, 5);
        ctx.fill();
        ctx.stroke();
      } else if (p.type === 'group') {
        ctx.fillStyle = '#7b3294';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(px, py - 7.0);
        ctx.lineTo(px + 7.0, py);
        ctx.lineTo(px, py + 7.0);
        ctx.lineTo(px - 7.0, py);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.fillStyle = p.type === 'obs' ? '#7f8c8d' : '#2c7fb8';
        ctx.arc(px, py, 6.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    ctx.restore();

    if (points.length <= 32) {
      const dots = points.map((p, idx) => ({ idx, px: tx(p.x), py: ty(p.y), label: _obsDisplayName(hms, p.dataset) }));
      const placedBoxes = [];
      const area = {
        l: Math.max(8, pad.l - 26),
        r: Math.min(cssW - 8, pad.l + pw + 26),
        t: Math.max(8, pad.t + 8),
        b: Math.min(cssH - 8, pad.t + ph - 8),
      };
      const labelH = 16;
      const boxGap = 2;
      const pointGap = 8;
      const intersects = (a, b, gap = boxGap) =>
        !(a.r + gap < b.l || b.r + gap < a.l || a.b + gap < b.t || b.b + gap < a.t);
      const overlapArea = (a, b) => {
        const w = Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l));
        const h = Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
        return w * h;
      };
      const makeBox = (x, y, align, labelW) => {
        let l = x, r = x;
        if (align === 'left') {
          r = x + labelW;
        } else if (align === 'right') {
          l = x - labelW;
        } else {
          l = x - labelW * 0.5;
          r = x + labelW * 0.5;
        }
        const t = y - labelH * 0.5;
        const b = y + labelH * 0.5;
        return { l, r, t, b };
      };
      const clampCandidate = (cand, labelW) => {
        let x = cand.x, y = cand.y;
        y = Math.min(Math.max(y, area.t + labelH * 0.5), area.b - labelH * 0.5);
        if (cand.align === 'left') {
          x = Math.min(Math.max(x, area.l), area.r - labelW);
        } else if (cand.align === 'right') {
          x = Math.max(Math.min(x, area.r), area.l + labelW);
        } else {
          x = Math.min(Math.max(x, area.l + labelW * 0.5), area.r - labelW * 0.5);
        }
        const box = makeBox(x, y, cand.align, labelW);
        return { x, y, align: cand.align, box };
      };

      ctx.fillStyle = '#2c3e50';
      ctx.font = '600 15px sans-serif';
      ctx.textBaseline = 'middle';

      dots.forEach(d => {
        const label = d.label;
        if (!label) return;
        const labelW = ctx.measureText(label).width;
        const baseCandidates = [
          { x: d.px + 11, y: d.py - 9, align: 'left' },
          { x: d.px + 11, y: d.py + 11, align: 'left' },
          { x: d.px - 11, y: d.py - 9, align: 'right' },
          { x: d.px - 11, y: d.py + 11, align: 'right' },
          { x: d.px, y: d.py - 14, align: 'center' },
          { x: d.px, y: d.py + 14, align: 'center' },
          { x: d.px + 20, y: d.py - 16, align: 'left' },
          { x: d.px + 20, y: d.py + 18, align: 'left' },
          { x: d.px - 20, y: d.py - 16, align: 'right' },
          { x: d.px - 20, y: d.py + 18, align: 'right' },
          { x: d.px + 28, y: d.py, align: 'left' },
          { x: d.px - 28, y: d.py, align: 'right' },
          { x: d.px, y: d.py - 24, align: 'center' },
          { x: d.px, y: d.py + 24, align: 'center' },
        ];
        const candidates = baseCandidates.map(c => clampCandidate(c, labelW));

        let chosen = null;
        for (const c of candidates) {
          if (placedBoxes.some(prev => intersects(c.box, prev))) continue;
          const overlapsPoint = dots.some(q => {
            if (q.idx === d.idx) return false;
            return q.px >= c.box.l - pointGap && q.px <= c.box.r + pointGap && q.py >= c.box.t - pointGap && q.py <= c.box.b + pointGap;
          });
          if (overlapsPoint) continue;
          chosen = c;
          break;
        }

        if (!chosen) {
          let best = null;
          let bestScore = Infinity;
          for (const c of candidates) {
            const overlapScore = placedBoxes.reduce((acc, b) => acc + overlapArea(c.box, b), 0);
            const pointHits = dots.reduce((acc, q) => {
              if (q.idx === d.idx) return acc;
              return acc + ((q.px >= c.box.l - pointGap && q.px <= c.box.r + pointGap && q.py >= c.box.t - pointGap && q.py <= c.box.b + pointGap) ? 1 : 0);
            }, 0);
            const dx = c.x - d.px, dy = c.y - d.py;
            const dist = Math.hypot(dx, dy);
            const score = overlapScore * 0.9 + pointHits * 140 + dist * 0.2;
            if (score < bestScore) {
              bestScore = score;
              best = c;
            }
          }
          chosen = best || candidates[0];
        }

        placedBoxes.push(chosen.box);
        let endX = chosen.x;
        if (chosen.align === 'left') endX = chosen.box.l;
        else if (chosen.align === 'right') endX = chosen.box.r;
        const lineDist = Math.hypot(endX - d.px, chosen.y - d.py);
        if (lineDist > 12) {
          ctx.strokeStyle = '#8a9fb4';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(d.px, d.py);
          ctx.lineTo(endX, chosen.y);
          ctx.stroke();
        }
        ctx.textAlign = chosen.align;
        ctx.fillText(label, chosen.x, chosen.y);
      });
    }

    ctx.fillStyle = '#22384d';
    ctx.font = '600 19px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(xLabel, pad.l + pw * 0.5, pad.t + ph + 40);
    ctx.save();
    ctx.translate(pad.l - 74, pad.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
  function _updateStats(hms, points) {
    const statEl = _query(document, `.cross-stat[data-hms="${hms}"][data-role="stats"]`);
    const extraEl = _query(document, `.cross-stat[data-hms="${hms}"][data-role="stats-extra"]`);
    if (!statEl || !extraEl) return;
    if (!points.length) {
      statEl.textContent = 'n=0';
      extraEl.textContent = '';
      return;
    }
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const r = _pearson(xs, ys);
    const rho = _spearman(xs, ys);
    const reg = _regression(xs, ys);
    const pval = _permPValue(xs, ys, 1200);
    const pth = _num(_query(document, `input[data-hms="${hms}"][data-role="p-thresh"]`)?.value);
    const threshold = Number.isFinite(pth) && pth > 0 ? pth : 0.05;
    const sig = Number.isFinite(pval) ? (pval < threshold ? 'significant' : 'not significant') : 'N/A';
    const regEq = (
      Number.isFinite(reg.slope) && Number.isFinite(reg.intercept)
        ? `y = ${_fmtSig(reg.slope, 3)}x ${reg.intercept >= 0 ? '+' : '-'} ${_fmtSig(Math.abs(reg.intercept), 3)}`
        : 'y = nan'
    );
    const q = { q1: 0, q2: 0, q3: 0, q4: 0 };
    points.forEach(p => {
      if (p.x >= 0 && p.y >= 0) q.q1 += 1;
      else if (p.x < 0 && p.y >= 0) q.q2 += 1;
      else if (p.x < 0 && p.y < 0) q.q3 += 1;
      else q.q4 += 1;
    });
    let influenceLine = '';
    if (points.length >= 4 && Number.isFinite(r)) {
      let maxDelta = -1, maxName = '';
      for (let i = 0; i < points.length; i++) {
        const xx = xs.filter((_, j) => j !== i);
        const yy = ys.filter((_, j) => j !== i);
        const rr = _pearson(xx, yy);
        if (Number.isFinite(rr)) {
          const d = Math.abs(rr - r);
          if (d > maxDelta) { maxDelta = d; maxName = _obsDisplayName(hms, points[i].dataset); }
        }
      }
      if (maxDelta >= 0) influenceLine = `Max leave-one-out Δr = ${maxDelta.toFixed(3)} (${maxName})`;
    }
    const statLines = [
      `n = ${points.length}`,
      `Pearson r = ${Number.isFinite(r) ? r.toFixed(3) : 'nan'}`,
      `Spearman ρ = ${Number.isFinite(rho) ? rho.toFixed(3) : 'nan'}`,
      `Regression = ${regEq}`,
      `p-value = ${Number.isFinite(pval) ? pval.toFixed(4) : 'nan'} (${sig})`,
    ];
    statEl.textContent = statLines.join('\n');
    const extraLines = [
      'Quadrant Counts',
      `Q1 = ${q.q1}`,
      `Q2 = ${q.q2}`,
      `Q3 = ${q.q3}`,
      `Q4 = ${q.q4}`,
    ];
    if (influenceLine) extraLines.push(influenceLine);
    extraEl.textContent = extraLines.join('\n');
  }
  function _pairPoints(hms) {
    const includeObs1 = !!_query(document, `input[data-hms="${hms}"][data-role="include-obs1"]`)?.checked;
    const includeObs2 = !!_query(document, `input[data-hms="${hms}"][data-role="include-obs2"]`)?.checked;
    const includeGroupMean = !!_query(document, `input[data-hms="${hms}"][data-role="include-groupmean"]`)?.checked;
    const modelState = _modelSelectionState(hms);
    const xr = _filteredRecords(hms, 'x', true);
    const yr = _filteredRecords(hms, 'y', true);
    const xMap = new Map();
    xr.forEach(r => {
      const name = String(r.dataset_name);
      const x = _valueNum(r.value);
      const obsKind = _recordObsKind(hms, r);
      const baseType = _recordDatasetType(hms, r);
      const type = obsKind ? 'obs' : baseType;
      if (!xMap.has(name)) {
        xMap.set(name, { x, type, obsKind });
        return;
      }
      const old = xMap.get(name) || { x: NaN, type: 'model', obsKind: '' };
      if (!Number.isFinite(old.x) && Number.isFinite(x)) old.x = x;
      if (!old.obsKind && obsKind) old.obsKind = obsKind;
      if (old.obsKind || type === 'obs') old.type = 'obs';
      else if (old.type !== 'obs' && type === 'group') old.type = 'group';
      xMap.set(name, old);
    });
    const yMap = new Map();
    yr.forEach(r => {
      const name = String(r.dataset_name);
      const y = _valueNum(r.value);
      const obsKind = _recordObsKind(hms, r);
      const baseType = _recordDatasetType(hms, r);
      const type = obsKind ? 'obs' : baseType;
      if (!yMap.has(name)) {
        yMap.set(name, { y, type, obsKind });
        return;
      }
      const old = yMap.get(name) || { y: NaN, type: 'model', obsKind: '' };
      if (!Number.isFinite(old.y) && Number.isFinite(y)) old.y = y;
      if (!old.obsKind && obsKind) old.obsKind = obsKind;
      if (old.obsKind || type === 'obs') old.type = 'obs';
      else if (old.type !== 'obs' && type === 'group') old.type = 'group';
      yMap.set(name, old);
    });
    const pts = [];
    xMap.forEach((xv, name) => {
      const yv = yMap.get(name);
      if (!yv) return;
      const x = _valueNum(xv.x), y = _valueNum(yv.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const obsKind = xv.obsKind || yv.obsKind || '';
      const type = (obsKind || xv.type === 'obs' || yv.type === 'obs')
        ? 'obs'
        : ((xv.type === 'group' || yv.type === 'group') ? 'group' : 'model');
      if (type === 'obs') {
        if (obsKind === 'obs1' && !includeObs1) return;
        if (obsKind === 'obs2' && !includeObs2) return;
        if (!obsKind && !(includeObs1 || includeObs2)) return;
      } else if (type === 'group') {
        if (!includeGroupMean) return;
      } else if (modelState.total > 0 && !modelState.selected.has(name)) {
        return;
      }
      pts.push({ dataset: name, type, obsKind, x, y });
    });
    pts.sort((a, b) => {
      if (a.type !== b.type) return a.type < b.type ? -1 : 1;
      return a.dataset < b.dataset ? -1 : (a.dataset > b.dataset ? 1 : 0);
    });
    return pts;
  }
  function _downloadCsv(hms, points) {
    const xLabel = _axisLabel(hms, 'x');
    const yLabel = _axisLabel(hms, 'y');
    const rows = [['dataset', 'type', 'x', 'y', 'x_label', 'y_label']];
    points.forEach(p => rows.push([p.dataset, p.type, String(p.x), String(p.y), xLabel, yLabel]));
    const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${hms}_cross_module_points.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1200);
  }
  function _downloadPng(hms) {
    const canvas = document.getElementById(`cross-canvas-${hms}`);
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(480, Math.round(rect.width || canvas.clientWidth || 860));
    const cssH = Math.max(480, Math.round(rect.height || canvas.clientHeight || cssW));
    const exportCanvas = document.createElement('canvas');
    _drawScatter(
      hms,
      _pairPoints(hms),
      _axisLabel(hms, 'x'),
      _axisLabel(hms, 'y'),
      {
        canvas: exportCanvas,
        cssW,
        cssH,
        dpr: _EXPORT_DPI / _CSS_DPI,
        skipButtonSync: true,
      }
    );
    const a = document.createElement('a');
    a.download = `${hms}_cross_module_scatter.png`;
    const clickDownload = (href, revoke) => {
      a.href = href;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        if (revoke) URL.revokeObjectURL(href);
        a.remove();
      }, 1200);
    };
    if (typeof exportCanvas.toBlob === 'function') {
      exportCanvas.toBlob(blob => {
        if (!blob) {
          clickDownload(exportCanvas.toDataURL('image/png'), false);
          return;
        }
        clickDownload(URL.createObjectURL(blob), true);
      }, 'image/png');
      return;
    }
    clickDownload(exportCanvas.toDataURL('image/png'), false);
  }
  function _render(hms) {
    _updateObsLabels(hms);
    const points = _pairPoints(hms);
    _updateStats(hms, points);
    _renderPointsTable(hms, points);
    _drawScatter(hms, points, _axisLabel(hms, 'x'), _axisLabel(hms, 'y'));
  }
  function _bindHms(hms) {
    const panel = _panel(hms);
    if (!panel) return;
    ['x', 'y'].forEach(axis => {
      ['module', 'coverage', 'view', 'region', 'domain', 'metric'].forEach(role => {
        const el = _axisEl(hms, axis, role);
        if (el) {
          el.addEventListener('change', () => {
            _refreshAxis(hms, axis);
            _render(hms);
          });
        }
      });
    });
    const obs1El = _query(document, `input[data-hms="${hms}"][data-role="include-obs1"]`);
    const obs2El = _query(document, `input[data-hms="${hms}"][data-role="include-obs2"]`);
    const groupEl = _query(document, `input[data-hms="${hms}"][data-role="include-groupmean"]`);
    if (obs1El) obs1El.addEventListener('change', () => _render(hms));
    if (obs2El) obs2El.addEventListener('change', () => _render(hms));
    if (groupEl) groupEl.addEventListener('change', () => _render(hms));
    const allModelEl = _query(document, `button[data-hms="${hms}"][data-role="models-all"]`);
    const noneModelEl = _query(document, `button[data-hms="${hms}"][data-role="models-none"]`);
    if (allModelEl) allModelEl.addEventListener('click', () => _setAllModels(hms, true));
    if (noneModelEl) noneModelEl.addEventListener('click', () => _setAllModels(hms, false));
    const pEl = _query(document, `input[data-hms="${hms}"][data-role="p-thresh"]`);
    if (pEl) pEl.addEventListener('change', () => _render(hms));
    const dlEl = _query(document, `button[data-hms="${hms}"][data-role="download-csv"]`);
    if (dlEl) dlEl.addEventListener('click', () => _downloadCsv(hms, _pairPoints(hms)));
    const pngEl = _query(document, `button[data-hms="${hms}"][data-role="export-png"]`);
    if (pngEl) pngEl.addEventListener('click', () => _downloadPng(hms));
    const navBtn = document.getElementById(`btn-${hms}-CrossModule`);
    if (navBtn) navBtn.addEventListener('click', () => setTimeout(() => _render(hms), 0));
    window.addEventListener('resize', () => _render(hms));
  }
  function _initHms(hms, recs) {
    const panel = _panel(hms);
    if (!panel) return;
    panel._records = Array.isArray(recs) ? recs : [];
    if (!panel._records.length) {
      const statEl = _query(document, `.cross-stat[data-hms="${hms}"][data-role="stats"]`);
      const extraEl = _query(document, `.cross-stat[data-hms="${hms}"][data-role="stats-extra"]`);
      if (statEl) statEl.textContent = 'No cross-module records for this hemisphere.';
      if (extraEl) extraEl.textContent = '';
      return;
    }
    _bindHms(hms);
    _renderModelSelector(hms);
    _refreshAxis(hms, 'x');
    _refreshAxis(hms, 'y');
    _render(hms);
  }
  function _extractRecords(payload) {
    return Array.isArray(payload && payload.records) ? payload.records : [];
  }
  function _inlinePayload() {
    const el = document.getElementById('cross-module-inline-data');
    if (!el) return null;
    const txt = String(el.textContent || '').trim();
    if (!txt) return null;
    try {
      return JSON.parse(txt);
    } catch (err) {
      return null;
    }
  }

  window.initCrossModuleExplorer = async function (jsonPath) {
    const path = String(jsonPath || 'cross_module_metrics.json');
    let payload = null;
    let fetchErr = null;
    if (path && typeof fetch === 'function') {
      try {
        const res = await fetch(path, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        payload = await res.json();
      } catch (err) {
        fetchErr = err;
      }
    }
    if (!payload) payload = _inlinePayload();
    if (payload) {
      const records = _extractRecords(payload);
      const nh = records.filter(r => String(r.hemisphere).toLowerCase() === 'nh');
      const sh = records.filter(r => String(r.hemisphere).toLowerCase() === 'sh');
      _initHms('nh', nh);
      _initHms('sh', sh);
      return;
    }
    ['nh', 'sh'].forEach(hms => {
      const statEl = _query(document, `.cross-stat[data-hms="${hms}"][data-role="stats"]`);
      const extraEl = _query(document, `.cross-stat[data-hms="${hms}"][data-role="stats-extra"]`);
      if (!statEl) return;
      if (fetchErr) {
        statEl.textContent = `Failed to load cross-module payload (${fetchErr && fetchErr.message ? fetchErr.message : fetchErr}).`;
      } else {
        statEl.textContent = 'No cross-module payload available.';
      }
      if (extraEl) extraEl.textContent = '';
    });
  };
  window.refreshCrossModuleExplorer = function (hms) {
    if (hms) {
      _render(String(hms).toLowerCase());
      return;
    }
    ['nh', 'sh'].forEach(_render);
  };
})();"""
    with open(cross_js_path, 'w', encoding='utf-8') as _fcj:
        _fcj.write(cross_js)

    # --- Full HTML document ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIToolv2 · {case_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2c3e50;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
header{{background:linear-gradient(135deg,#1a3a5c,#2980b9);color:#fff;padding:16px 28px;display:flex;align-items:baseline;gap:14px;flex-shrink:0}}
header h1{{font-size:1.35em;font-weight:600}}
.ts{{margin-left:auto;font-size:.78em;opacity:.75}}
/* Hemisphere tabs (top-level navigation) */
.hms-tabs{{display:flex;background:#16304a;padding:0 20px;gap:4px;flex-shrink:0}}
.hms-tab{{padding:11px 24px;background:none;border:none;border-bottom:3px solid transparent;
  color:#a8bfd0;cursor:pointer;font-size:.92em;font-weight:500;transition:.15s}}
.hms-tab:hover{{color:#fff;background:rgba(255,255,255,.06)}}
.hms-tab.active{{color:#7ec8e3;border-bottom-color:#7ec8e3;font-weight:700}}
/* Hemisphere panes */
.hms-pane{{display:none;flex:1;flex-direction:column;min-height:0}}
.hms-pane.active{{display:flex}}
/* Empty hemisphere placeholder */
.empty-hms{{display:flex;align-items:center;justify-content:center;flex:1;padding:60px 40px}}
.empty-hms p{{color:#7a8fa6;font-size:1em;font-style:italic;text-align:center;
  background:#fff;border-radius:8px;padding:24px 36px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
/* Module layout inside a hemisphere pane */
.layout{{display:flex;flex:1;min-height:0}}
nav{{width:160px;background:#1e2d3d;padding:0;flex-shrink:0;overflow-y:auto;min-height:0}}
.nav-btn{{display:block;width:100%;padding:11px 18px;background:none;border:none;
  color:#a8bfd0;text-align:left;cursor:pointer;font-size:.88em;
  border-left:3px solid transparent;transition:.15s}}
.nav-btn:hover{{background:#243b55;color:#fff}}
.nav-btn.active{{background:#243b55;color:#7ec8e3;border-left-color:#7ec8e3;font-weight:600}}
main{{flex:1;padding:0 28px 22px 28px;overflow-y:auto;min-height:0}}
.app-shell{{display:flex;flex:1;flex-direction:column;min-height:0}}
/* Panel layout */
.panel{{display:none;flex-direction:column;gap:0}}
.panel.active{{display:flex}}
/* Breadcrumb header */
.panel-header{{background:#e8eef5;padding:8px 16px;
  margin-bottom:20px;border-radius:4px;font-size:.85em;position:sticky;top:0;z-index:40;display:none}}
.bc-hms{{color:#5a7a9a;font-weight:400}}
.bc-sep{{margin:0 7px;color:#aab;font-size:1.1em}}
.bc-mod{{color:#1a3a5c;font-weight:700;letter-spacing:.02em}}
/* Section headings */
.section-title{{font-size:.92em;font-weight:700;color:#1a3a5c;text-transform:uppercase;
  letter-spacing:.06em;border-bottom:2px solid #2980b9;padding-bottom:5px;margin-bottom:16px}}
.metrics-section{{margin-bottom:30px}}
.metrics-section + .plots-section{{margin-top:34px}}
.plots-section{{margin-bottom:42px}}
.plots-section + .plots-section{{margin-top:38px;padding-top:18px;border-top:1px dashed #d2dde8}}
/* Figure grid */
.fig-grid{{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start}}
figure{{background:#fff;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.09);
  overflow:hidden;max-width:500px;flex:1 1 320px}}
.img-link{{display:block;cursor:zoom-in}}
figure img{{width:100%;display:block;transition:opacity .15s}}
.img-link:hover img{{opacity:.85}}
figcaption{{padding:7px 12px;font-size:.78em;color:#666;background:#fafafa;border-top:1px solid #eee}}
.geo-note{{margin-bottom:12px;color:#47627f;font-size:.84em}}
.geo-map-wrap{{display:flex;justify-content:center}}
.geo-figure{{max-width:600px!important;flex:1 1 auto!important}}
/* Lightbox */
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:9999;
  align-items:center;justify-content:center;cursor:zoom-out}}
#lb.open{{display:flex}}
#lb img{{max-width:95vw;max-height:95vh;border-radius:4px;box-shadow:0 4px 32px rgba(0,0,0,.6)}}
#lb-close{{position:fixed;top:16px;right:22px;color:#fff;font-size:2em;cursor:pointer;
  line-height:1;user-select:none;opacity:.8}}
#lb-close:hover{{opacity:1}}
/* Plot tuner side panel */
/* Scalar metrics table — CMIP evaluation paper style */
.tbl-wrap{{overflow-x:auto;margin-bottom:4px}}
.tbl-downloads{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;margin:4px 0 10px}}
.tbl-dl-btn{{display:inline-block;padding:4px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:12px;cursor:pointer;font-size:.78em;transition:.12s;text-decoration:none}}
.tbl-dl-btn:hover{{background:#e6f1fa}}
.tbl-download-hint{{width:100%;margin-top:2px;font-size:.72em;color:#6b7f92;line-height:1.3;text-align:right}}
table{{border-collapse:collapse;width:100%;font-size:.82em;border:1px solid #b0bec5}}
th,td{{padding:6px 12px;text-align:right;border:1px solid #cfd8dc}}
tr.col-header th{{background:#1a3a5c;color:#fff;font-weight:700;
  letter-spacing:.03em;text-align:center}}
tr.col-header th.sortable{{cursor:pointer;user-select:none;position:relative;padding-right:18px}}
tr.col-header th.sortable::after{{content:'↕';position:absolute;right:6px;top:50%;transform:translateY(-50%);opacity:.55;font-size:.9em}}
tr.col-header th.sortable.asc::after{{content:'↑';opacity:.95}}
tr.col-header th.sortable.desc::after{{content:'↓';opacity:.95}}
tr.col-header th:first-child{{text-align:left}}
tr.unit-row th{{background:#2c5282;color:#c8dff0;font-style:italic;
  font-weight:400;font-size:.88em;text-align:center}}
tr.unit-row th:first-child{{text-align:left}}
tbody tr:nth-child(odd){{background:#ffffff}}
tbody tr:nth-child(even){{background:#eef2f7}}
tbody tr:hover{{background:#dce8f5}}
tr.obs-row{{background:#dce8f0!important;font-style:italic}}
tr.obs-row td:first-child{{font-weight:600;color:#1a3a5c}}
.sicb-dadt{{font-weight:400}}

.siconc-region-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 10px}}
.siconc-region-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.siconc-region-tab:hover{{background:#e6f1fa}}
.siconc-region-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.siconc-region-pane{{display:none}}
.siconc-region-pane.active{{display:block}}
.region-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 10px}}
.region-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.region-tab:hover{{background:#e6f1fa}}
.region-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.region-pane{{display:none}}
.region-pane.active{{display:block}}
.region-coverage-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.region-coverage-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.region-coverage-tab:hover{{background:#e6f1fa}}
.region-coverage-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.region-coverage-pane{{display:none}}
.region-coverage-pane.active{{display:block}}
.region-season-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.region-season-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.region-season-tab:hover{{background:#e6f1fa}}
.region-season-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.region-season-pane{{display:none}}
.region-season-pane.active{{display:block}}
.region-phase-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.region-phase-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.region-phase-tab:hover{{background:#e6f1fa}}
.region-phase-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.region-phase-pane{{display:none}}
.region-phase-pane.active{{display:block}}
.subsection-title{{font-size:.83em;color:#2c5282;margin:10px 0 6px}}
.siconc-period-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.siconc-period-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.siconc-period-tab:hover{{background:#e6f1fa}}
.siconc-period-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.siconc-period-pane{{display:none}}
.siconc-period-pane.active{{display:block}}
.sicb-season-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.sicb-season-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.sicb-season-tab:hover{{background:#e6f1fa}}
.sicb-season-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.sicb-season-pane{{display:none}}
.sicb-season-pane.active{{display:block}}
.sitrans-phase-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.sitrans-phase-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.sitrans-phase-tab:hover{{background:#e6f1fa}}
.sitrans-phase-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.sitrans-phase-pane{{display:none}}
.sitrans-phase-pane.active{{display:block}}
td:first-child{{text-align:left;font-weight:400}}
/* Raw / Differences view tabs */
.view-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}}
.top-view-tabs{{margin:0}}
.view-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.view-tab:hover{{background:#e6f1fa}}
.view-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.view-pane{{display:none}}
.view-pane.active{{display:block}}
.module-controls{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;
  margin:0 0 14px;position:sticky;top:0;z-index:35;background:#f0f2f5;padding:8px 0 10px;
  border-bottom:1px solid #d8e2eb}}
.module-view-center{{display:flex;justify-content:center;justify-self:center;min-width:0}}
.module-controls-spacer{{min-width:1px}}
.module-view-tabs{{display:flex}}
.coverage-tabs-placeholder,.view-tabs-placeholder{{visibility:hidden;pointer-events:none}}
.coverage-tabs-placeholder .coverage-tab,.view-tabs-placeholder .view-tab{{pointer-events:none}}
/* Original / Obs-matched coverage tabs */
.coverage-section{{position:relative}}
.coverage-controls{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;
  margin:0 0 14px;position:sticky;top:0;z-index:35;background:#f0f2f5;padding:8px 0 10px;
  border-bottom:1px solid #d8e2eb}}
.coverage-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:0;justify-self:start}}
.coverage-view-center{{display:flex;justify-content:center;justify-self:center;min-width:0}}
.coverage-view-tabs{{display:none}}
.coverage-view-tabs.active{{display:flex}}
.coverage-controls-spacer{{min-width:1px}}
.coverage-tab{{padding:5px 10px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;
  border-radius:14px;cursor:pointer;font-size:.8em;transition:.12s}}
.coverage-tab:hover{{background:#e6f1fa}}
.coverage-tab.active{{background:#2c5282;color:#fff;border-color:#2c5282;font-weight:600}}
.coverage-pane{{display:none}}
.coverage-pane.active{{display:block}}
.coverage-pane .plots-section{{margin-top:38px;padding-top:18px;border-top:1px dashed #c8d4e1}}

/* Cross-module explorer */
.cross-module-section{{margin-top:14px}}
.cross-note{{margin:0 0 10px;color:#4a6078;font-size:.84em;line-height:1.4}}
.cross-layout{{display:grid;grid-template-columns:minmax(0,1fr);grid-template-areas:"axes" "main";row-gap:14px;align-items:start;margin:8px 0 14px}}
.cross-axis-row{{grid-area:axes;display:grid;grid-template-columns:240px 220px 220px 350px 350px;gap:12px;align-items:stretch;justify-content:center}}
.cross-main{{grid-area:main;min-width:0;display:flex;flex-direction:column;align-items:center}}
.cross-axis,.cross-opts,.cross-table-panel,.cross-model-filter{{background:#fff;border:1px solid #d7e0ea;border-radius:8px;padding:13px 15px;min-height:420px;display:flex;flex-direction:column}}
.cross-axis h4,.cross-opts h4,.cross-table-panel h4,.cross-model-filter h4{{font-size:1.14em;color:#1f4469;margin:0 0 10px}}
.cross-axis label,.cross-opts label{{display:block;font-size:.98em;color:#506780;margin:9px 0 6px}}
.cross-inline-check{{display:flex!important;align-items:center;gap:6px;margin:4px 0 0 0!important}}
.cross-inline-check input{{margin:0}}
.cross-axis select,.cross-opts input[type=\"number\"]{{width:100%;padding:8px 10px;border:1px solid #bac9d8;border-radius:6px;background:#fbfdff;font-size:1.02em;color:#2f4f69}}
.cross-opts .tbl-dl-btn,.cross-table-panel .tbl-dl-btn{{margin-top:12px;font-size:.92em;padding:7px 11px;align-self:flex-start}}
.cross-model-actions{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}}
.cross-mini-btn{{padding:4px 9px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;border-radius:12px;cursor:pointer;font-size:.78em;transition:.12s}}
.cross-mini-btn:hover{{background:#e6f1fa}}
.cross-model-count{{margin-left:auto;font-size:.86em;color:#506780}}
.cross-model-list{{flex:1 1 auto;min-height:0;overflow:auto;border:1px solid #d7e0ea;border-radius:8px;padding:7px 8px;background:#fbfdff}}
.cross-model-item{{display:flex;align-items:flex-start;gap:7px;font-size:.92em;color:#35526b;line-height:1.35;margin:3px 0;cursor:pointer;word-break:break-word}}
.cross-model-item input{{margin-top:2px;flex:0 0 auto}}
.cross-model-empty{{font-size:.88em;color:#7a8ea3;padding:6px 2px}}
.cross-stat{{margin-top:10px;font-size:1.01em;color:#304d67;line-height:1.55;white-space:pre-line}}
.cross-plot-wrap{{position:relative;display:block;background:#fff;border:1px solid #d7e0ea;border-radius:8px;padding:8px 8px 40px;width:min(82vw,816px);max-width:100%;margin:0 auto}}
#cross-canvas-nh,#cross-canvas-sh{{width:100%;height:auto;aspect-ratio:1 / 1;display:block}}
.cross-export-btn{{position:absolute;right:12px;bottom:10px;padding:7px 11px;border:1px solid #90a4ae;background:#f8fbff;color:#35526b;border-radius:14px;cursor:pointer;font-size:.84em;transition:.12s}}
.cross-export-btn:hover{{background:#e6f1fa}}
.cross-table-wrap{{margin-top:8px;flex:1 1 auto;min-height:0;overflow:auto;border:1px solid #d7e0ea;border-radius:8px;background:#fff;width:100%}}
.cross-table{{width:100%;border-collapse:collapse;font-size:.84em}}
.cross-table th,.cross-table td{{border:1px solid #dde5ef;padding:4px 5px;text-align:right;white-space:nowrap}}
.cross-table th:first-child,.cross-table td:first-child,.cross-table th:nth-child(2),.cross-table td:nth-child(2){{text-align:left}}
.cross-table thead th{{position:sticky;top:0;background:#1a3a5c;color:#fff;z-index:2}}
@media (max-width: 1580px) {{
  .cross-axis-row{{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}}
  .cross-axis-row .cross-opts,.cross-axis-row .cross-table-panel{{grid-column:1 / -1}}
}}
@media (max-width: 980px) {{
  .cross-axis-row{{grid-template-columns:1fr}}
}}
</style>
</head>
<body data-case="{case_name}">
<header>
  <h1>SIToolv2 — {case_name}</h1>
  <span class="ts">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</header>
<div class="hms-tabs">
{hms_tabs_html}
</div>
<div class="app-shell">
{panes_html}
</div>
<!-- Lightbox overlay: click any thumbnail to view full resolution -->
<div id="lb" role="dialog" aria-modal="true" aria-label="Full-resolution image viewer">
  <span id="lb-close" title="Close (Esc)">&#x2715;</span>
  <img id="lb-img" src="" alt="">
</div>
<script src="report_download.js"></script>
<script id="cross-module-inline-data" type="application/json">{cross_payload_inline_json}</script>
<script id="cross-module-obs-names" type="application/json">{obs_name_map_inline_json}</script>
<script src="report_cross_module.js"></script>
<script>
// Switch the active hemisphere tab and show its pane
function showHms(hms){{
  document.querySelectorAll('.hms-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.hms-tab').forEach(b=>b.classList.remove('active'));
  const pane=document.getElementById('hms-'+hms);
  if(pane)pane.classList.add('active');
  const tab=document.querySelector('.hms-tab[data-hms="'+hms+'"]');
  if(tab)tab.classList.add('active');
}}

// Switch the visible module panel within a hemisphere and highlight its nav button
function showMod(hms,mod){{
  const pane=document.getElementById('hms-'+hms);
  if(!pane)return;
  pane.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  pane.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  const panel=document.getElementById('tab-'+hms+'-'+mod);
  const btn=document.getElementById('btn-'+hms+'-'+mod);
  if(panel)panel.classList.add('active');
  if(btn)btn.classList.add('active');
  if(mod==='CrossModule' && typeof window.refreshCrossModuleExplorer==='function'){{
    window.refreshCrossModuleExplorer(hms);
  }}
}}

// Return currently active module-level coverage mode (base/matched), if any.
function getCoverageMode(hms, mod){{
  const activeCoveragePane=document.querySelector('.coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"].active');
  return activeCoveragePane ? activeCoveragePane.dataset.mode : null;
}}

// Switch generic region table within one hemisphere/module
function showRegionTable(hms, mod, region, preferredMode=null){{
  const coveragePaneCount=document.querySelectorAll('.coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"]').length;
  let mode=preferredMode;
  if(!mode && coveragePaneCount>0){{
    mode=getCoverageMode(hms, mod) || 'base';
  }}
  const scopedSel = mode ? '[data-cov="'+mode+'"]' : '';
  document.querySelectorAll('.region-pane[data-hms="'+hms+'"][data-mod="'+mod+'"]'+scopedSel).forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.region-tab[data-hms="'+hms+'"][data-mod="'+mod+'"]'+scopedSel).forEach(b=>b.classList.remove('active'));
  let panes=Array.from(document.querySelectorAll('.region-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'+scopedSel));
  if(!panes.length && mode){{
    panes=Array.from(document.querySelectorAll('.region-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'));
  }}
  panes.forEach(p=>p.classList.add('active'));
  if(panes.length){{
    const pane=panes[0];
    const baseCoverageBtn=pane.querySelector('.region-coverage-tab[data-mode="base"]');
    if(baseCoverageBtn){{
      showRegionCoverage(hms, mod, region, 'base');
    }}else{{
      const firstCoverageBtn=pane.querySelector('.region-coverage-tab');
      if(firstCoverageBtn){{
        showRegionCoverage(hms, mod, region, firstCoverageBtn.dataset.mode);
      }}
    }}

    const springBtn=pane.querySelector('.region-season-tab[data-season="Spring"]');
    if(springBtn){{
      showRegionSeason(hms, mod, region, 'Spring', mode);
    }}else{{
      const firstSeasonBtn=pane.querySelector('.region-season-tab');
      if(firstSeasonBtn){{
        showRegionSeason(hms, mod, region, firstSeasonBtn.dataset.season, mode);
      }}
    }}

    const advanceBtn=pane.querySelector('.region-phase-tab[data-phase="Advance"]');
    if(advanceBtn){{
      showRegionPhase(hms, mod, region, 'Advance', mode);
    }}else{{
      const firstPhaseBtn=pane.querySelector('.region-phase-tab');
      if(firstPhaseBtn){{
        showRegionPhase(hms, mod, region, firstPhaseBtn.dataset.phase, mode);
      }}
    }}
  }}
  let tabs=Array.from(document.querySelectorAll('.region-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'+scopedSel));
  if(!tabs.length && mode){{
    tabs=Array.from(document.querySelectorAll('.region-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'));
  }}
  tabs.forEach(tab=>tab.classList.add('active'));
}}

// Switch coverage mode within one region table
function showRegionCoverage(hms, mod, region, mode){{
  document.querySelectorAll('.region-coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.region-coverage-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]').forEach(b=>b.classList.remove('active'));
  const pane=document.querySelector('.region-coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-mode="'+mode+'"]');
  if(pane)pane.classList.add('active');
  const tab=document.querySelector('.region-coverage-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-mode="'+mode+'"]');
  if(tab)tab.classList.add('active');
}}

// Switch seasonal table within one region (SICB regional payload)
function showRegionSeason(hms, mod, region, season, preferredMode=null){{
  const coveragePaneCount=document.querySelectorAll('.coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"]').length;
  let mode=preferredMode;
  if(!mode && coveragePaneCount>0){{
    mode=getCoverageMode(hms, mod) || 'base';
  }}
  const scopedSel = mode ? '[data-cov="'+mode+'"]' : '';
  document.querySelectorAll('.region-season-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'+scopedSel).forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.region-season-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'+scopedSel).forEach(b=>b.classList.remove('active'));
  let panes=Array.from(document.querySelectorAll('.region-season-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-season="'+season+'"]'+scopedSel));
  if(!panes.length && mode){{
    panes=Array.from(document.querySelectorAll('.region-season-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-season="'+season+'"]'));
  }}
  panes.forEach(p=>p.classList.add('active'));
  let tabs=Array.from(document.querySelectorAll('.region-season-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-season="'+season+'"]'+scopedSel));
  if(!tabs.length && mode){{
    tabs=Array.from(document.querySelectorAll('.region-season-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-season="'+season+'"]'));
  }}
  tabs.forEach(tab=>tab.classList.add('active'));
}}

// Switch phase table within one region (SItrans regional payload)
function showRegionPhase(hms, mod, region, phase, preferredMode=null){{
  const coveragePaneCount=document.querySelectorAll('.coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"]').length;
  let mode=preferredMode;
  if(!mode && coveragePaneCount>0){{
    mode=getCoverageMode(hms, mod) || 'base';
  }}
  const scopedSel = mode ? '[data-cov="'+mode+'"]' : '';
  document.querySelectorAll('.region-phase-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'+scopedSel).forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.region-phase-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"]'+scopedSel).forEach(b=>b.classList.remove('active'));
  let panes=Array.from(document.querySelectorAll('.region-phase-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-phase="'+phase+'"]'+scopedSel));
  if(!panes.length && mode){{
    panes=Array.from(document.querySelectorAll('.region-phase-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-phase="'+phase+'"]'));
  }}
  panes.forEach(p=>p.classList.add('active'));
  let tabs=Array.from(document.querySelectorAll('.region-phase-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-phase="'+phase+'"]'+scopedSel));
  if(!tabs.length && mode){{
    tabs=Array.from(document.querySelectorAll('.region-phase-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-region="'+region+'"][data-phase="'+phase+'"]'));
  }}
  tabs.forEach(tab=>tab.classList.add('active'));
}}


// Switch SIconc region table within one hemisphere
function showSiconcRegion(hms, region){{
  document.querySelectorAll('.siconc-region-pane[data-hms="'+hms+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.siconc-region-tab[data-hms="'+hms+'"]').forEach(b=>b.classList.remove('active'));
  const pane=document.querySelector('.siconc-region-pane[data-hms="'+hms+'"][data-region="'+region+'"]');
  if(pane)pane.classList.add('active');
  const tab=document.querySelector('.siconc-region-tab[data-hms="'+hms+'"][data-region="'+region+'"]');
  if(tab)tab.classList.add('active');

  const annualBtn=document.querySelector('.siconc-period-tab[data-hms="'+hms+'"][data-region="'+region+'"][data-period="Annual"]');
  if(annualBtn){{
    showSiconcPeriod(hms, region, 'Annual');
    return;
  }}
  const firstBtn=document.querySelector('.siconc-period-tab[data-hms="'+hms+'"][data-region="'+region+'"]');
  if(firstBtn){{
    showSiconcPeriod(hms, region, firstBtn.dataset.period);
  }}
}}

// Switch SIconc period table within one hemisphere + region
function showSiconcPeriod(hms, region, period){{
  let regionKey=region;
  let periodKey=period;
  if(periodKey===undefined){{
    // Backward compatibility: showSiconcPeriod(hms, period)
    regionKey='__default__';
    periodKey=region;
  }}
  document.querySelectorAll('.siconc-period-pane[data-hms="'+hms+'"][data-region="'+regionKey+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.siconc-period-tab[data-hms="'+hms+'"][data-region="'+regionKey+'"]').forEach(b=>b.classList.remove('active'));
  const pane=document.querySelector('.siconc-period-pane[data-hms="'+hms+'"][data-region="'+regionKey+'"][data-period="'+periodKey+'"]');
  if(pane)pane.classList.add('active');
  const tab=document.querySelector('.siconc-period-tab[data-hms="'+hms+'"][data-region="'+regionKey+'"][data-period="'+periodKey+'"]');
  if(tab)tab.classList.add('active');
}}

// Switch SICB seasonal table within one hemisphere
function showSicbSeason(hms, season){{
  document.querySelectorAll('.sicb-season-pane[data-hms="'+hms+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.sicb-season-tab[data-hms="'+hms+'"]').forEach(b=>b.classList.remove('active'));
  const pane=document.getElementById('sicb-season-'+hms+'-'+season);
  if(pane)pane.classList.add('active');
  const tab=document.querySelector('.sicb-season-tab[data-hms="'+hms+'"][data-season="'+season+'"]');
  if(tab)tab.classList.add('active');
}}

// Switch SItrans phase table within one hemisphere
function showSitransPhase(hms, phase){{
  document.querySelectorAll('.sitrans-phase-pane[data-hms="'+hms+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.sitrans-phase-tab[data-hms="'+hms+'"]').forEach(b=>b.classList.remove('active'));
  const pane=document.getElementById('sitrans-phase-'+hms+'-'+phase);
  if(pane)pane.classList.add('active');
  const tab=document.querySelector('.sitrans-phase-tab[data-hms="'+hms+'"][data-phase="'+phase+'"]');
  if(tab)tab.classList.add('active');
}}

// Switch Raw / Differences view within a module panel section
function showView(hms, mod, group, view){{
  document.querySelectorAll('.view-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-group="'+group+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.view-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-group="'+group+'"]').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.view-pane[data-hms="'+hms+'"][data-mod="'+mod+'"][data-group="'+group+'"][data-view="'+view+'"]').forEach(p=>p.classList.add('active'));
  document.querySelectorAll('.view-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-group="'+group+'"][data-view="'+view+'"]').forEach(tab=>tab.classList.add('active'));
}}

// Switch Original / Obs-Matched coverage mode within a module panel
function showCoverage(hms, mod, mode){{
  document.querySelectorAll('.coverage-pane[data-hms="'+hms+'"][data-mod="'+mod+'"]').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.coverage-tab[data-hms="'+hms+'"][data-mod="'+mod+'"]').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.coverage-view-tabs[data-hms="'+hms+'"][data-mod="'+mod+'"]').forEach(v=>v.classList.remove('active'));
  const pane=document.getElementById('coverage-'+hms+'-'+mod+'-'+mode);
  if(pane)pane.classList.add('active');
  document.querySelectorAll('.coverage-tab[data-hms="'+hms+'"][data-mod="'+mod+'"][data-mode="'+mode+'"]').forEach(tab=>tab.classList.add('active'));
  const activeViewTabs=document.querySelector('.coverage-view-tabs[data-hms="'+hms+'"][data-mod="'+mod+'"][data-cov="'+mode+'"]');
  if(activeViewTabs){{
    activeViewTabs.classList.add('active');
    const selectedBtn=activeViewTabs.querySelector('.view-tab.active')
      || activeViewTabs.querySelector('.view-tab[data-view="raw"]')
      || activeViewTabs.querySelector('.view-tab');
    if(selectedBtn){{
      showView(hms, mod, mode, selectedBtn.dataset.view || 'raw');
    }}
  }}
  if(pane){{
    const firstRegionTab=pane.querySelector('.region-tab[data-cov="'+mode+'"]') || pane.querySelector('.region-tab');
    if(firstRegionTab){{
      showRegionTable(hms, mod, firstRegionTab.dataset.region, mode);
    }}
  }}
}}

// Initialise: show active hemisphere and first module for each hemisphere that has content
showHms('{active_hemisphere}');
{init_mods_js}
if (typeof window.initCrossModuleExplorer === 'function') {{
  window.initCrossModuleExplorer('cross_module_metrics.json');
}}
const seenRegionModules=new Set();
document.querySelectorAll('.region-tab').forEach(btn=>{{
  const key=btn.dataset.hms+'|'+btn.dataset.mod+'|'+(btn.dataset.cov||'single');
  if(seenRegionModules.has(key))return;
  seenRegionModules.add(key);
  showRegionTable(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, btn.dataset.cov || null);
}});
const seenRegionCoverage=new Set();
document.querySelectorAll('.region-coverage-tab').forEach(btn=>{{
  const key=btn.dataset.hms+'|'+btn.dataset.mod+'|'+btn.dataset.region;
  if(seenRegionCoverage.has(key))return;
  seenRegionCoverage.add(key);
  const baseBtn=document.querySelector('.region-coverage-tab[data-hms="'+btn.dataset.hms+'"][data-mod="'+btn.dataset.mod+'"][data-region="'+btn.dataset.region+'"][data-mode="base"]');
  if(baseBtn){{
    showRegionCoverage(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, 'base');
  }}else{{
    showRegionCoverage(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, btn.dataset.mode);
  }}
}});
const seenRegionSeason=new Set();
document.querySelectorAll('.region-season-tab').forEach(btn=>{{
  const key=btn.dataset.hms+'|'+btn.dataset.mod+'|'+btn.dataset.region+'|'+(btn.dataset.cov||'single');
  if(seenRegionSeason.has(key))return;
  seenRegionSeason.add(key);
  const springBtn=document.querySelector('.region-season-tab[data-hms="'+btn.dataset.hms+'"][data-mod="'+btn.dataset.mod+'"][data-region="'+btn.dataset.region+'"][data-season="Spring"]');
  if(springBtn){{
    showRegionSeason(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, 'Spring', btn.dataset.cov || null);
  }}else{{
    showRegionSeason(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, btn.dataset.season, btn.dataset.cov || null);
  }}
}});
const seenRegionPhase=new Set();
document.querySelectorAll('.region-phase-tab').forEach(btn=>{{
  const key=btn.dataset.hms+'|'+btn.dataset.mod+'|'+btn.dataset.region+'|'+(btn.dataset.cov||'single');
  if(seenRegionPhase.has(key))return;
  seenRegionPhase.add(key);
  let sel='.region-phase-tab[data-hms="'+btn.dataset.hms+'"][data-mod="'+btn.dataset.mod+'"][data-region="'+btn.dataset.region+'"]';
  if(btn.dataset.cov) sel+='[data-cov="'+btn.dataset.cov+'"]';
  const advanceBtn=document.querySelector(sel+'[data-phase="Advance"]');
  if(advanceBtn){{
    showRegionPhase(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, 'Advance', btn.dataset.cov || null);
  }}else{{
    showRegionPhase(btn.dataset.hms, btn.dataset.mod, btn.dataset.region, btn.dataset.phase, btn.dataset.cov || null);
  }}
}});
const seenSiconcHms=new Set();
document.querySelectorAll('.siconc-region-tab').forEach(btn=>{{
  const hms=btn.dataset.hms;
  if(seenSiconcHms.has(hms))return;
  seenSiconcHms.add(hms);
  showSiconcRegion(hms, btn.dataset.region);
}});
// Backward-compatible init for legacy non-regional SIconc payloads
document.querySelectorAll('.siconc-period-tab[data-region="__default__"][data-period="Annual"]').forEach(btn=>{{
  showSiconcPeriod(btn.dataset.hms, '__default__', 'Annual');
}});
document.querySelectorAll('.sicb-season-tab[data-season="Spring"]').forEach(btn=>{{
  showSicbSeason(btn.dataset.hms, 'Spring');
}});
document.querySelectorAll('.sitrans-phase-tab[data-phase="Advance"]').forEach(btn=>{{
  showSitransPhase(btn.dataset.hms, 'Advance');
}});
// Initialise coverage tabs: default to "Original Coverage"
document.querySelectorAll('.coverage-tab[data-mode="base"]').forEach(btn=>{{
  showCoverage(btn.dataset.hms, btn.dataset.mod, 'base');
}});
// Initialise Raw/Diff view tabs: default to "Raw Values"
document.querySelectorAll('.view-tab[data-view="raw"]').forEach(btn=>{{
  showView(btn.dataset.hms, btn.dataset.mod, btn.dataset.group, 'raw');
}});

// Generic sortable tables for Scalar Metrics Summary
function toSortValue(txt){{
  const t=(txt||'').trim();
  const n=Number(t);
  return Number.isFinite(n) ? n : t.toLowerCase();
}}
function sortMetricTable(table, colIdx, th){{
  const tbody=table.querySelector('tbody');
  if(!tbody)return;
  const rows=Array.from(tbody.querySelectorAll('tr'));
  const curr=th.dataset.sortDir||'';
  const next=curr==='asc'?'desc':'asc';

  table.querySelectorAll('thead tr.col-header th.sortable').forEach(h=>{{
    h.classList.remove('asc','desc');
    h.dataset.sortDir='';
  }});
  th.classList.add(next);
  th.dataset.sortDir=next;

  const obsRows=[];
  const dataRows=[];
  rows.forEach(r=>{{(r.classList.contains('obs-row')?obsRows:dataRows).push(r);}});

  dataRows.sort((a,b)=>{{
    const av=toSortValue(a.cells[colIdx]?.textContent||'');
    const bv=toSortValue(b.cells[colIdx]?.textContent||'');
    if(typeof av==='number' && typeof bv==='number'){{
      return next==='asc' ? av-bv : bv-av;
    }}
    if(av<bv)return next==='asc'?-1:1;
    if(av>bv)return next==='asc'?1:-1;
    return 0;
  }});

  [...obsRows, ...dataRows].forEach(r=>tbody.appendChild(r));
}}

document.querySelectorAll('.metrics-section table').forEach(table=>{{
  table.querySelectorAll('thead tr.col-header th.sortable').forEach((th, idx)=>{{
    th.addEventListener('click',()=>sortMetricTable(table, idx, th));
  }});
}});

// Lightbox: open on thumbnail click, close on overlay click or Esc key
const lb=document.getElementById('lb');
const lbImg=document.getElementById('lb-img');
document.querySelectorAll('.img-link').forEach(a=>{{
  a.addEventListener('click',e=>{{
    e.preventDefault();
    lbImg.src=a.dataset.src;
    lb.classList.add('open');
  }});
}});
function closeLb(){{lb.classList.remove('open');lbImg.src='';}}
document.getElementById('lb-close').addEventListener('click',closeLb);
lb.addEventListener('click',e=>{{if(e.target===lb)closeLb();}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeLb();}});
</script>
</body>
</html>"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info("HTML report → %s", report_path)
