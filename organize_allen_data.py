#!/usr/bin/env python3
"""Organize brain-region counts using Allen Brain Atlas metadata and summaries.

This script loads an Excel workbook containing per-brain region counts, enriches
regions with Allen Brain Atlas metadata (structure tree lookups only), and
produces:

* A detailed CSV with per-entry annotations and percentage contributions.
* An aggregated CSV summarizing each region with totals, mean percentage across
  brains, and standard error of the mean (SEM).
* An Excel workbook collecting the key summary tables (per-region, parent
  groups, thalamic groups, and dorsal thalamus polymodal subdivisions, each with
  mean % and SEM, plus per-brain breakdowns).
* Five percentage bar charts (mean ± SEM) covering cortical vs subcortical
  composition, cortical mid-level subdivisions, major subcortical groups,
  thalamic groups, and dorsal thalamus polymodal subdivisions.
* A text file logging how workbook region names were matched to Allen
  structures (exact, override, approximate, unresolved).

Default input is ``Book1.xlsx``. Pass a different workbook via ``--input``.
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from allensdk.core.structure_tree import StructureTree

DEFAULT_INPUT = "Book1.xlsx"
DEFAULT_MANIFEST = "allen_manifest.json"
DEFAULT_OUTPUT_DIR = "results"

CUSTOM_NAME_OVERRIDES: Dict[str, str] = {
    "Globus pallidus": "Globus pallidus, external segment",
    "Periventricular hypothalamic nucleus": "Periventricular hypothalamic nucleus, intermediate part",
    "Mediodorsal nucleus of the thalamus": "Mediodorsal nucleus of thalamus",
    "Superior colliculus": "Superior colliculus, sensory related",
    "Substantia nigra": "Substantia nigra, compact part",
}

PRIMARY_GROUP_PATTERNS: List[Tuple[str, Tuple[Tuple[str, ...], ...]]] = [
    ("Hypothalamus", (("hypothalamus",),)),
    ("Thalamus", (("thalamus",),)),
    ("Striatum", (("striatum",),)),
    ("Pallidum", (("pallidum",),)),
    ("Septal area", (("septal",),)),
    ("Hippocampus", (("hippocampus",),)),
    ("Amygdala", (("amygdala",),)),
    ("Midbrain", (("midbrain",),)),
    ("Cerebellum", (("cerebellum",),)),
    ("Brain stem", (("brain", "stem"), ("hindbrain",),)),
    ("Olfactory areas", (("olfactory",),)),
    ("Basal forebrain", (("basal", "forebrain"), ("substantia", "innominata"))),
]


@dataclass
class AllenStructureMetadata:
    structure_id: int
    acronym: str
    name: str
    parent_name: Optional[str]
    major_division: Optional[str]
    cortical_class: Optional[str]
    is_cortical: bool
    is_subcortical: bool
    primary_group: str
    thalamic_group: Optional[str]
    thalamic_subdivision: Optional[str]
    path_names: List[str]


@dataclass
class StructureLookup:
    tree: StructureTree
    name_to_structure: Dict[str, Dict]
    names: List[str]

    def resolve(self, region: str) -> Tuple[Optional[Dict], Optional[str], str]:
        """Return (structure dict, resolved name, match type)."""
        if region in self.name_to_structure:
            return self.name_to_structure[region], region, "exact"

        if region in CUSTOM_NAME_OVERRIDES:
            target = CUSTOM_NAME_OVERRIDES[region]
            structure = self._fetch_structure(target)
            logging.warning("Region '%s' mapped via manual override to '%s'", region, target)
            return structure, target, "override"

        candidates = difflib.get_close_matches(region, self.names, n=5, cutoff=0.75)
        match = self._choose_candidate(region, candidates, require_base=True)
        if match is None:
            relaxed = self._choose_candidate(region, candidates, require_base=False)
            if relaxed and self._first_keyword_present(region, relaxed):
                match = relaxed
        if match is None:
            substring_hits = [name for name in self.names if region.lower() in name.lower()]
            match = self._choose_candidate(region, substring_hits, require_base=False)

        if match is None:
            logging.warning("Region '%s' not found in Allen structure tree", region)
            return None, None, "unresolved"

        structure = self._fetch_structure(match)
        logging.warning("Region '%s' approximated with Allen structure '%s'", region, match)
        return structure, match, "approx"

    def _choose_candidate(self, region: str, candidates: Sequence[str], require_base: bool) -> Optional[str]:
        if not candidates:
            return None
        base = region.lower().split(",")[0].strip()
        prioritized = []
        for cand in candidates:
            cand_lower = cand.lower()
            has_base = bool(base) and base in cand_lower
            starts_with_base = bool(base) and cand_lower.startswith(base)
            has_comma = "," in cand
            prioritized.append((not has_base, not starts_with_base, not has_comma, len(cand), cand))
        prioritized.sort()
        if require_base and prioritized[0][0]:
            return None
        return prioritized[0][4]

    def _first_keyword_present(self, region: str, candidate: str) -> bool:
        tokens = [t for t in region.lower().replace("/", " ").split() if t not in {"of", "the", "and"}]
        if not tokens:
            return True
        first = tokens[0]
        return first in candidate.lower()

    def _fetch_structure(self, name: str) -> Dict:
        if name not in self.name_to_structure:
            structure = self.tree.get_structures_by_name([name])[0]
            self.name_to_structure[name] = structure
            self.names.append(name)
        return self.name_to_structure[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help="Excel workbook with brain region counts (default: Book1.xlsx).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST),
        help="Manifest JSON used by the Allen SDK for structure lookups only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory where CSVs, Excel summaries, figures, and notes are written.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("allen_region_summary.csv"),
        help="Filename for the aggregated region summary CSV (relative to output dir).",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=Path("allen_region_details.csv"),
        help="Filename for the detailed annotated CSV (relative to output dir).",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=Path("allen_summary.xlsx"),
        help="Filename for the Excel workbook collecting summary tables (relative to output dir).",
    )
    parser.add_argument(
        "--mapping-notes",
        type=Path,
        default=Path("mapping_notes.txt"),
        help="Filename for the mapping notes text file (relative to output dir).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s [%(levelname)s] %(message)s")


def load_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    logging.info("Loading workbook %s", path)
    excel = pd.ExcelFile(path)
    frames: List[pd.DataFrame] = []

    for sheet in excel.sheet_names:
        if sheet.lower().startswith("export"):
            continue
        frame = excel.parse(sheet)
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
        frame.columns = [c.strip() if isinstance(c, str) else c for c in frame.columns]
        frame = frame.rename(columns={"n": "count", "normalised": "normalized"})
        frame["source_sheet"] = sheet
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Region"])

    for col in ("count", "normalized"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    logging.info("Loaded %d rows across %d sheets", len(combined), len(frames))
    return combined


def get_structure_tree(manifest: Path) -> StructureTree:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cache = ReferenceSpaceCache(
        resolution=25,
        reference_space_key="annotation/ccf_2017",
        manifest=str(manifest),
    )
    return cache.get_structure_tree()


def determine_primary_group(path_names: Sequence[str], is_cortical: bool) -> str:
    if is_cortical:
        return "Cortex"

    tokens: List[str] = []
    for name in path_names:
        cleaned = name.replace(',', ' ').replace('/', ' ').lower()
        tokens.extend(cleaned.split())
    token_set = set(tokens)

    for label, pattern_sets in PRIMARY_GROUP_PATTERNS:
        for pattern in pattern_sets:
            if all(token in token_set for token in pattern):
                return label
    return "Other"


def build_metadata(tree: StructureTree, structure: Dict, resolved_name: str) -> AllenStructureMetadata:
    path_structures = tree.get_structures_by_id(structure["structure_id_path"])
    path_names = [item["name"] for item in path_structures]

    parent_name = path_names[-2] if len(path_names) > 1 else None
    major_division = next((name for name in path_names if name in {"Cerebrum", "Brain stem", "Cerebellum"}), None)
    cortical_markers = {"Isocortex", "Hippocampal formation", "Olfactory areas", "Retrohippocampal region"}
    cortical_class = next((name for name in path_names if name in cortical_markers), None)
    is_cortical = cortical_class is not None

    subcortical_markers = {"Cerebral nuclei", "Striatum", "Pallidum", "Thalamus", "Hypothalamus", "Midbrain", "Hindbrain"}
    is_subcortical = any(name in path_names for name in subcortical_markers) and not is_cortical

    thalamic_group = None
    thalamic_subdivision = None
    if "Thalamus" in path_names:
        idx = path_names.index("Thalamus")
        if idx + 1 < len(path_names):
            thalamic_group = path_names[idx + 1]
        if idx + 2 < len(path_names):
            candidate = path_names[idx + 1]
            if "dorsal" in candidate.lower():
                thalamic_subdivision = candidate
            else:
                thalamic_subdivision = path_names[idx + 2]

    primary_group = determine_primary_group(path_names, is_cortical)

    return AllenStructureMetadata(
        structure_id=structure["id"],
        acronym=structure["acronym"],
        name=resolved_name,
        parent_name=parent_name,
        major_division=major_division,
        cortical_class=cortical_class,
        is_cortical=is_cortical,
        is_subcortical=is_subcortical,
        primary_group=primary_group,
        thalamic_group=thalamic_group,
        thalamic_subdivision=thalamic_subdivision,
        path_names=path_names,
    )


def attach_allen_annotations(tree: StructureTree, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = tree.nodes()
    lookup = StructureLookup(
        tree=tree,
        name_to_structure={node["name"]: node for node in nodes},
        names=[node["name"] for node in nodes],
    )

    metadata_records: List[Dict[str, object]] = []
    mapping_records: List[Dict[str, object]] = []

    for region in df["Region"].drop_duplicates():
        structure, resolved_name, match_type = lookup.resolve(region)
        mapping_records.append(
            {
                "Region": region,
                "matched_name": resolved_name,
                "mapping_type": match_type,
            }
        )
        if structure is None or resolved_name is None:
            continue
        meta = build_metadata(lookup.tree, structure, resolved_name)
        metadata_records.append(
            {
                "Region": region,
                "allen_id": meta.structure_id,
                "allen_acronym": meta.acronym,
                "allen_name": meta.name,
                "allen_parent": meta.parent_name,
                "allen_major_division": meta.major_division,
                "allen_cortical_class": meta.cortical_class,
                "is_cortical": meta.is_cortical,
                "is_subcortical": meta.is_subcortical,
                "allen_primary_group": meta.primary_group,
                "thalamic_group": meta.thalamic_group,
                "thalamic_subdivision": meta.thalamic_subdivision,
                "allen_path": json.dumps(meta.path_names),
                "mapping_type": match_type,
            }
        )

    meta_df = pd.DataFrame(metadata_records)
    merged = df.merge(meta_df, on="Region", how="left")

    missing = merged[merged["allen_id"].isna()]["Region"].unique()
    if len(missing):
        logging.warning("Allen metadata missing for %d regions", len(missing))

    mapping_df = pd.DataFrame(mapping_records)
    return merged, mapping_df


def add_percentages(df: pd.DataFrame) -> pd.DataFrame:
    if "count" not in df.columns:
        df["count"] = 0
    df["count"] = df["count"].fillna(0)

    total = df["count"].sum()
    df["count_pct_total"] = (df["count"] / total) * 100 if total else 0

    if "source_sheet" in df.columns:
        brain_totals = df.groupby("source_sheet")["count"].transform("sum")
        df["count_pct_brain"] = df["count"] / brain_totals.replace(0, pd.NA)
        df["count_pct_brain"] = df["count_pct_brain"].fillna(0) * 100

        # Add thalamus-specific percentages
        thalamus_totals = df[df["allen_primary_group"] == "Thalamus"].groupby("source_sheet")["count"].transform("sum")
        df["count_pct_thalamus"] = 0.0
        thalamus_mask = df["allen_primary_group"] == "Thalamus"
        if thalamus_mask.any():
            df.loc[thalamus_mask, "count_pct_thalamus"] = (df.loc[thalamus_mask, "count"] / thalamus_totals).fillna(0) * 100
    else:
        df["count_pct_brain"] = 0
        df["count_pct_thalamus"] = 0

    return df


def aggregate_regions(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {col: "sum" for col in ("count", "normalized") if col in df.columns}
    if "source_sheet" in df.columns:
        aggregations["source_sheet"] = lambda rows: ",".join(sorted(set(map(str, rows))))

    summary = (
        df.groupby("Region", as_index=False)
        .agg(aggregations)
        .merge(
            df.drop_duplicates("Region")[
                [
                    col
                    for col in [
                        "Region",
                        "allen_id",
                        "allen_acronym",
                        "allen_name",
                        "allen_parent",
                        "allen_major_division",
                        "allen_cortical_class",
                        "allen_primary_group",
                        "is_cortical",
                        "is_subcortical",
                        "thalamic_group",
                        "thalamic_subdivision",
                        "allen_path",
                        "mapping_type",
                    ]
                    if col in df.columns
                ]
            ],
            on="Region",
            how="left",
        )
    )

    if "count" not in summary.columns:
        summary["count"] = 0
    summary["count"] = summary["count"].fillna(0)

    total_count = summary["count"].sum()
    summary["count_pct_total"] = (summary["count"] / total_count) * 100 if total_count else 0

    ordered_cols = [
        "Region",
        "allen_name",
        "allen_id",
        "allen_acronym",
        "allen_parent",
        "allen_major_division",
        "allen_cortical_class",
        "allen_primary_group",
        "is_cortical",
        "is_subcortical",
        "thalamic_group",
        "thalamic_subdivision",
        "mapping_type",
        "count",
        "normalized",
        "count_pct_total",
        "source_sheet",
        "allen_path",
    ]
    summary = summary[[col for col in ordered_cols if col in summary.columns]]
    return summary.sort_values("Region").reset_index(drop=True)


def compute_region_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_brain = (
        df.groupby(["Region", "allen_name", "source_sheet"], dropna=False)["count_pct_brain"]
        .sum()
        .reset_index()
    )
    stats = (
        per_brain.groupby(["Region", "allen_name"], dropna=False)["count_pct_brain"]
        .agg(mean_pct="mean", sem_pct="sem", brain_count="count")
        .reset_index()
    )
    return stats, per_brain


def compute_group_stats_from_annotated(
    annotated_subset: pd.DataFrame,
    group_col: str,
    categories: Optional[Iterable[str]] = None,
    use_thalamus_pct: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pct_col = 'count_pct_thalamus' if use_thalamus_pct else 'count_pct_brain'
    if annotated_subset.empty or 'source_sheet' not in annotated_subset.columns or pct_col not in annotated_subset.columns:
        stats_columns = [group_col, 'mean_pct', 'sem_pct', 'brain_count']
        per_brain_columns = [group_col, 'source_sheet', pct_col]
        return pd.DataFrame(columns=stats_columns), pd.DataFrame(columns=per_brain_columns)

    data = annotated_subset.dropna(subset=['source_sheet']).copy()
    data = data.loc[data[group_col].notna()]
    if data.empty:
        stats_columns = [group_col, 'mean_pct', 'sem_pct', 'brain_count']
        per_brain_columns = [group_col, 'source_sheet', pct_col]
        return pd.DataFrame(columns=stats_columns), pd.DataFrame(columns=per_brain_columns)

    brains = sorted(data['source_sheet'].unique())
    if not brains:
        stats_columns = [group_col, 'mean_pct', 'sem_pct', 'brain_count']
        per_brain_columns = [group_col, 'source_sheet', pct_col]
        return pd.DataFrame(columns=stats_columns), pd.DataFrame(columns=per_brain_columns)

    if categories is None:
        categories = sorted(data[group_col].unique())
    else:
        categories = [cat for cat in categories if cat is not None]
        if not categories:
            categories = sorted(data[group_col].unique())
        else:
            categories = list(dict.fromkeys(categories))

    index = pd.MultiIndex.from_product([categories, brains], names=[group_col, 'source_sheet'])
    grouped = (
        data.groupby([group_col, 'source_sheet'])[pct_col]
        .sum()
        .reindex(index, fill_value=0)
        .reset_index()
    )

    stats = (
        grouped.groupby(group_col)[pct_col]
        .agg(mean_pct='mean', sem_pct='sem')
        .reset_index()
    )
    stats['sem_pct'] = stats['sem_pct'].fillna(0.0)
    stats['brain_count'] = len(brains)
    stats = stats.sort_values('mean_pct', ascending=False).reset_index(drop=True)

    return stats, grouped


def summarize_by_group(
    summary: pd.DataFrame,
    annotated: pd.DataFrame,
    group_col: str,
    summary_filter: Optional[pd.Series] = None,
    use_thalamus_pct: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in summary.columns or group_col not in annotated.columns:
        return pd.DataFrame(), pd.DataFrame()

    if summary_filter is not None:
        summary_subset = summary.loc[summary_filter].copy()
    else:
        summary_subset = summary.copy()

    summary_subset = summary_subset.loc[summary_subset[group_col].notna()]
    if summary_subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    allowed_regions = summary_subset['Region'].unique()
    annotated_subset = annotated.loc[annotated['Region'].isin(allowed_regions) & annotated[group_col].notna()].copy()
    if annotated_subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    count_totals = summary_subset.groupby(group_col)['count'].sum().reset_index(name='total_count')
    total_all = count_totals['total_count'].sum()
    count_totals['pct_total_count'] = (count_totals['total_count'] / total_all) * 100 if total_all else 0

    categories = count_totals[group_col].tolist()
    stats, per_brain = compute_group_stats_from_annotated(annotated_subset, group_col, categories=categories, use_thalamus_pct=use_thalamus_pct)
    if stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = count_totals.merge(stats, on=group_col, how='left')
    merged = merged.sort_values('mean_pct', ascending=False).reset_index(drop=True)
    return merged, per_brain


def compute_cortical_subcortical_stats(annotated: pd.DataFrame) -> pd.DataFrame:
    if annotated.empty:
        return pd.DataFrame(columns=['category', 'mean_pct', 'sem_pct', 'brain_count'])

    data = annotated[['source_sheet', 'count_pct_brain', 'is_cortical', 'is_subcortical']].dropna(subset=['source_sheet']).copy()
    if data.empty:
        return pd.DataFrame(columns=['category', 'mean_pct', 'sem_pct', 'brain_count'])

    def label_row(row: pd.Series) -> str:
        if bool(row.get('is_cortical')):
            return 'Cortex'
        if bool(row.get('is_subcortical')):
            return 'Subcortical'
        return None  # Skip mixed/other entries

    data['category'] = data.apply(label_row, axis=1)
    # Filter out None/mixed entries
    data = data.dropna(subset=['category'])
    stats, _ = compute_group_stats_from_annotated(
        data[['source_sheet', 'count_pct_brain', 'category']],
        'category',
        categories=['Cortex', 'Subcortical'],
    )
    stats = stats.loc[(stats['mean_pct'] > 0) | (stats['sem_pct'] > 0)].reset_index(drop=True)
    return stats


def get_custom_colors(categories: List[str], category_col: str, title: str = "") -> List[str]:
    """Get custom color palette for different anatomical categories."""

    # Blue to Green to Orange gradient palette for variety
    blue_colors = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5']  # Blues
    green_colors = ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']  # Greens
    orange_colors = ['#E65100', '#F57C00', '#FF9800', '#FFB74D', '#FFCC02']  # Oranges

    # Combined gradient palette: blue → green → orange
    gradient_palette = blue_colors + green_colors + orange_colors

    def assign_colors_with_thalamus_priority(cats: List[str]) -> List[str]:
        """Assign colors ensuring any category with 'thalamus' gets green."""
        colors = []
        green_index = 0
        non_thalamus_index = 0

        for cat in cats:
            if 'thalamus' in cat.lower():
                # Always assign green shades to thalamus
                colors.append(green_colors[green_index % len(green_colors)])
                green_index += 1
            else:
                # Use blue or orange for non-thalamus regions
                if non_thalamus_index < len(blue_colors):
                    colors.append(blue_colors[non_thalamus_index])
                else:
                    colors.append(orange_colors[(non_thalamus_index - len(blue_colors)) % len(orange_colors)])
                non_thalamus_index += 1
        return colors

    if category_col == 'category':  # Cortical vs Subcortical
        color_map = {
            'Cortex': '#2E86AB',      # Bright blue
            'Subcortical': '#E76F51'   # Coral/orange - contrasting
        }
        return [color_map.get(cat, '#777777') for cat in categories]
    elif 'thalamic' in category_col.lower() or 'thalamus' in category_col.lower() or 'dorsal_polymodal' in category_col.lower():
        # Pure thalamic analyses - use only green palette
        return green_colors[:len(categories)]
    elif 'parent' in category_col.lower() or 'cortical' in category_col.lower():
        # Mixed groups that might contain thalamus - use smart assignment
        return assign_colors_with_thalamus_priority(categories)
    else:
        # Default to gradient palette for other groups
        return gradient_palette[:len(categories)]


def plot_stats_bar(stats: pd.DataFrame, category_col: str, output_path: Path, title: str) -> None:
    if stats.empty or category_col not in stats.columns or 'mean_pct' not in stats.columns:
        logging.warning("Skipping plot '%s' due to missing data", title)
        return

    data = stats.loc[stats[category_col].notna()].copy()
    data['mean_pct'] = data['mean_pct'].astype(float)
    data['sem_pct'] = data['sem_pct'].fillna(0.0).astype(float)
    data = data.loc[(data['mean_pct'] > 0) | (data['sem_pct'] > 0)]
    if data.empty:
        logging.warning("No data to plot for '%s'", title)
        return

    data = data.sort_values('mean_pct', ascending=True)
    y_positions = range(len(data))
    colors = get_custom_colors(data[category_col].tolist(), category_col, title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(4, len(data) * 0.4)))
    ax = plt.gca()
    ax.barh(y_positions, data['mean_pct'], xerr=data['sem_pct'], color=colors, edgecolor='black', capsize=6)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(data[category_col])
    # Set appropriate axis label based on analysis type
    if 'thalamic' in category_col.lower() or 'thalamus' in title.lower():
        ax.set_xlabel('Mean percentage of thalamus (%)')
    else:
        ax.set_xlabel('Mean percentage of brain (%)')
    ax.set_title(title)
    max_x = float((data['mean_pct'] + data['sem_pct']).max()) if not data.empty else 0.0
    ax.set_xlim(0, max(max_x * 1.1, 1.0))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Saved figure %s", output_path)


def generate_figures(
    annotated: pd.DataFrame,
    parent_summary: pd.DataFrame,
    cortical_midlevel_summary: pd.DataFrame,
    thalamus_summary: pd.DataFrame,
    dorsal_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / 'figures'

    cortical_stats = compute_cortical_subcortical_stats(annotated)
    plot_stats_bar(cortical_stats, 'category', figures_dir / 'cortical_vs_subcortical.png', 'Cortical vs subcortical (mean ± SEM)')

    if not cortical_midlevel_summary.empty:
        plot_stats_bar(
            cortical_midlevel_summary,
            'cortical_parent',
            figures_dir / 'cortical_midlevel_subdivisions.png',
            'Cortical parent subdivisions (mean ± SEM)',
        )
    else:
        logging.warning('No cortical parent data available for mid-level plot')

    subcortical_stats = parent_summary.loc[parent_summary['parent_group'] != 'Cortex'].copy()
    if not subcortical_stats.empty:
        plot_stats_bar(
            subcortical_stats,
            'parent_group',
            figures_dir / 'subcortical_subdivisions.png',
            'Subcortical parent groups (mean ± SEM)',
        )
    else:
        logging.warning('No subcortical parent groups available for plotting')

    if not thalamus_summary.empty:
        plot_stats_bar(
            thalamus_summary,
            'thalamic_group',
            figures_dir / 'thalamus_groups.png',
            'Thalamic groups (mean ± SEM)',
        )
    else:
        logging.warning('No thalamic group data available for plotting')

    if not dorsal_summary.empty:
        plot_stats_bar(
            dorsal_summary,
            'dorsal_polymodal_subdivision',
            figures_dir / 'dorsal_thalamus_polymodal.png',
            'Dorsal thalamus polymodal subdivisions (mean ± SEM)',
        )
    else:
        logging.warning('No polymodal thalamic subdivision data available for plotting')



def export_excel_summary(
    output_path: Path,
    summary: pd.DataFrame,
    region_stats: pd.DataFrame,
    per_brain_region: pd.DataFrame,
    parent_summary: pd.DataFrame,
    parent_per_brain: pd.DataFrame,
    cortical_midlevel_summary: pd.DataFrame,
    cortical_midlevel_per_brain: pd.DataFrame,
    thalamus_summary: pd.DataFrame,
    thalamus_per_brain: pd.DataFrame,
    dorsal_summary: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="region_summary", index=False)
        region_stats.to_excel(writer, sheet_name="region_mean_sem", index=False)
        per_brain_region.to_excel(writer, sheet_name="region_per_brain_pct", index=False)
        parent_summary.to_excel(writer, sheet_name="parent_groups", index=False)
        parent_per_brain.to_excel(writer, sheet_name="parent_per_brain", index=False)
        cortical_midlevel_summary.to_excel(writer, sheet_name="cortical_midlevel", index=False)
        cortical_midlevel_per_brain.to_excel(writer, sheet_name="cortical_midlevel_per_brain", index=False)
        thalamus_summary.to_excel(writer, sheet_name="thalamus_groups", index=False)
        thalamus_per_brain.to_excel(writer, sheet_name="thalamus_per_brain", index=False)
        dorsal_summary.to_excel(writer, sheet_name="dorsal_polymodal", index=False)
        mapping_df.to_excel(writer, sheet_name="mapping_notes", index=False)
    logging.info("Wrote Excel summary workbook to %s", output_path)



def write_mapping_notes(path: Path, mapping_df: pd.DataFrame) -> None:
    lines = []
    for _, row in mapping_df.sort_values("Region").iterrows():
        region = row["Region"]
        matched = row.get("matched_name")
        mtype = row.get("mapping_type")
        if pd.isna(matched):
            lines.append(f"Region '{region}' -> no Allen match (mapping_type={mtype})")
        else:
            lines.append(f"Region '{region}' -> '{matched}' (mapping_type={mtype})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logging.info("Wrote mapping notes to %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_workbook(args.input)
    tree = get_structure_tree(args.manifest)
    annotated, mapping_df = attach_allen_annotations(tree, df)
    annotated = add_percentages(annotated)

    details_path = output_dir / args.details_csv
    annotated.to_csv(details_path, index=False)
    logging.info("Wrote detailed rows to %s", details_path)

    summary = aggregate_regions(annotated)
    region_stats, per_brain_region = compute_region_statistics(annotated)
    summary = summary.merge(region_stats, on=["Region", "allen_name"], how="left")

    summary_path = output_dir / args.summary_csv
    summary.to_csv(summary_path, index=False)
    logging.info("Wrote region summary to %s", summary_path)

    # Group-level summaries
    parent_summary, parent_per_brain = summarize_by_group(summary, annotated, "allen_primary_group")
    if not parent_summary.empty:
        parent_summary = parent_summary.rename(columns={"allen_primary_group": "parent_group"})
    else:
        parent_summary = pd.DataFrame(columns=["parent_group", "total_count", "pct_total_count", "mean_pct", "sem_pct", "brain_count"])
    if not parent_per_brain.empty:
        parent_per_brain = parent_per_brain.rename(columns={"allen_primary_group": "parent_group"})
    else:
        parent_per_brain = pd.DataFrame(columns=["parent_group", "source_sheet", "count_pct_brain"])

    cortical_filter = summary["is_cortical"] == True
    cortical_midlevel_summary, cortical_midlevel_per_brain = summarize_by_group(summary, annotated, "allen_parent", summary_filter=cortical_filter)
    if not cortical_midlevel_summary.empty:
        cortical_midlevel_summary = cortical_midlevel_summary.rename(columns={"allen_parent": "cortical_parent"})
    else:
        cortical_midlevel_summary = pd.DataFrame(columns=["cortical_parent", "total_count", "pct_total_count", "mean_pct", "sem_pct", "brain_count"])
    if not cortical_midlevel_per_brain.empty:
        cortical_midlevel_per_brain = cortical_midlevel_per_brain.rename(columns={"allen_parent": "cortical_parent"})
    else:
        cortical_midlevel_per_brain = pd.DataFrame(columns=["cortical_parent", "source_sheet", "count_pct_brain"])

    thalamus_mask = summary["allen_primary_group"] == "Thalamus"
    thalamus_summary, thalamus_per_brain = summarize_by_group(summary, annotated, "thalamic_group", summary_filter=thalamus_mask, use_thalamus_pct=True)
    if thalamus_summary.empty:
        thalamus_summary = pd.DataFrame(columns=["thalamic_group", "total_count", "pct_total_count", "mean_pct", "sem_pct", "brain_count"])
    if thalamus_per_brain.empty:
        thalamus_per_brain = pd.DataFrame(columns=["thalamic_group", "source_sheet", "count_pct_brain"])

    dorsal_mask = summary["thalamic_group"].notna() & summary["thalamic_group"].str.contains("polymodal", case=False, na=False)
    dorsal_summary, _ = summarize_by_group(summary, annotated, "thalamic_subdivision", summary_filter=dorsal_mask, use_thalamus_pct=True)
    if not dorsal_summary.empty:
        dorsal_summary = dorsal_summary.rename(columns={"thalamic_subdivision": "dorsal_polymodal_subdivision"})
    else:
        dorsal_summary = pd.DataFrame(columns=["dorsal_polymodal_subdivision", "total_count", "pct_total_count", "mean_pct", "sem_pct", "brain_count"])

    # Export Excel workbook
    excel_path = output_dir / args.excel_output
    export_excel_summary(
        excel_path,
        summary,
        region_stats,
        per_brain_region,
        parent_summary,
        parent_per_brain,
        cortical_midlevel_summary,
        cortical_midlevel_per_brain,
        thalamus_summary,
        thalamus_per_brain,
        dorsal_summary,
        mapping_df,
    )

    # Mapping notes text file
    mapping_path = output_dir / args.mapping_notes
    write_mapping_notes(mapping_path, mapping_df)

    # Figures
    generate_figures(
        annotated,
        parent_summary,
        cortical_midlevel_summary,
        thalamus_summary,
        dorsal_summary,
        output_dir,
    )


if __name__ == "__main__":
    main()
