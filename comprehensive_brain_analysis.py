#!/usr/bin/env python3
"""
Comprehensive Brain Analysis Script

This script generates 11 different graphs and comprehensive Excel files
based on the user's specifications for brain region analysis.

Graphs:
1. Cortical vs Subcortical regions
2. All subcortical regions (striatum, thalamus, hypothalamus, pallidum, amygdala, midbrain, other)
3. All cortical regions (somatosensory, somatomotor, isocortex, olfactory, auditory, hippocampus)
4. Thalamus zoom: DORpm and DORsm
5. DORpm subdivisions
6. DORsm subdivisions  
7. Intralaminar nuclei
8. Isocortex subdivisions
9. Basal ganglia subdivisions
10. Heat map: All regions per brain
11. Heat map: Bigger regions per brain

All graphs are normalized to 100% within their categories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("/Users/hiro/Documents/yana-data")
RESULTS_DIR = DATA_DIR / "comprehensive_results"
FIGURES_DIR = RESULTS_DIR / "figures"
EXCEL_DIR = RESULTS_DIR / "excel"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
EXCEL_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the detailed brain region data"""
    details_path = DATA_DIR / "results" / "allen_region_details.csv"
    summary_path = DATA_DIR / "results" / "allen_region_summary.csv"
    
    details_df = pd.read_csv(details_path)
    summary_df = pd.read_csv(summary_path)
    
    return details_df, summary_df

def filter_minimum_cells(df, min_cells=2, group_col='Region'):
    """Filter regions with minimum number of cells across all brains"""
    region_counts = df.groupby(group_col)['count'].sum()
    valid_regions = region_counts[region_counts >= min_cells].index
    
    filtered_df = df[df[group_col].isin(valid_regions)].copy()
    
    # Create excluded regions log
    excluded = region_counts[region_counts < min_cells]
    if len(excluded) > 0:
        excluded_log = RESULTS_DIR / "excluded_regions_min_cells.txt"
        with open(excluded_log, 'w') as f:
            f.write("Regions excluded due to minimum cell threshold (<2 cells):\n")
            f.write("=" * 60 + "\n")
            for region, count in excluded.items():
                f.write(f"{region}: {count} cells\n")
        print(f"Excluded {len(excluded)} regions with <2 cells. See {excluded_log}")
    
    return filtered_df

def calculate_group_stats(df, group_col, value_col='count_pct_brain', min_brains=3):
    """Calculate mean and SEM for grouped data"""
    # Filter for regions present in minimum number of brains
    brain_counts = df.groupby(group_col)['source_sheet'].nunique()
    valid_groups = brain_counts[brain_counts >= min_brains].index
    filtered_df = df[df[group_col].isin(valid_groups)].copy()
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Calculate statistics
    stats = filtered_df.groupby(group_col)[value_col].agg([
        ('mean_pct', 'mean'),
        ('sem_pct', lambda x: x.std() / np.sqrt(len(x))),
        ('brain_count', 'count'),
        ('total_cells', 'sum')
    ]).reset_index()
    
    # Normalize to 100% within the group
    total_mean = stats['mean_pct'].sum()
    if total_mean > 0:
        stats['mean_pct_normalized'] = (stats['mean_pct'] / total_mean) * 100
        stats['sem_pct_normalized'] = (stats['sem_pct'] / total_mean) * 100
    else:
        stats['mean_pct_normalized'] = 0
        stats['sem_pct_normalized'] = 0
    
    return stats.sort_values('mean_pct_normalized', ascending=False)

def create_bar_plot(stats, title, output_path, xlabel="Mean percentage (%)", figsize=(12, 8)):
    """Create a horizontal bar plot with error bars"""
    if stats.empty:
        print(f"No data to plot for {title}")
        return
    
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(stats))
    
    # Create bars
    bars = plt.barh(y_pos, stats['mean_pct_normalized'], 
                    xerr=stats['sem_pct_normalized'], 
                    capsize=5, alpha=0.7)
    
    # Color bars based on category
    colors = plt.cm.Set3(np.linspace(0, 1, len(stats)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(y_pos, stats.iloc[:, 0])  # First column should be the group name
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(stats['mean_pct_normalized'], stats['sem_pct_normalized'])):
        plt.text(mean + sem + 1, i, f'{mean:.1f}±{sem:.1f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save as PNG, SVG, and PDF
    base_path = output_path.with_suffix('')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.svg", bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {base_path}.png, .svg, .pdf")

def create_heatmap(data, title, output_path, figsize=(12, 8)):
    """Create a heatmap with proper normalization"""
    plt.figure(figsize=figsize)
    
    # Create pivot table for heatmap
    if 'Region' in data.columns and 'source_sheet' in data.columns:
        pivot_data = data.pivot_table(
            values='count_pct_brain', 
            index='Region', 
            columns='source_sheet', 
            fill_value=0
        )
    else:
        # For grouped data
        group_col = [col for col in data.columns if col not in ['source_sheet', 'count_pct_brain', 'mean_pct', 'sem_pct']][0]
        pivot_data = data.pivot_table(
            values='count_pct_brain',
            index=group_col,
            columns='source_sheet',
            fill_value=0
        )
    
    # Normalize each brain to 100% within its category (cortex or subcortex)
    pivot_data = pivot_data.div(pivot_data.sum(axis=0), axis=1) * 100
    
    # Create heatmap with better formatting
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', 
                cbar_kws={'label': 'Percentage (%)'}, 
                annot_kws={'size': 8})
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Brain', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save as PNG, SVG, and PDF
    base_path = output_path.with_suffix('')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.svg", bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {base_path}.png, .svg, .pdf")

def create_excel_sheets(details_df, summary_df):
    """Create comprehensive Excel file with all data"""
    excel_path = EXCEL_DIR / "comprehensive_brain_analysis.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Raw data summary
        summary_df.to_excel(writer, sheet_name='Raw_Data_Summary', index=False)
        
        # Sheet 2: Cortical vs Subcortical
        cortical_mask = details_df['is_cortical'] == True
        subcortical_mask = details_df['is_subcortical'] == True
        
        cortical_data = details_df[cortical_mask]
        subcortical_data = details_df[subcortical_mask]
        
        cortical_stats = calculate_group_stats(cortical_data, 'is_cortical')
        subcortical_stats = calculate_group_stats(subcortical_data, 'is_subcortical')
        
        cortical_vs_subcortical = pd.concat([
            pd.DataFrame({'Category': ['Cortical'], 'mean_pct_normalized': [cortical_data['count_pct_brain'].sum()], 
                         'sem_pct_normalized': [cortical_data['count_pct_brain'].std() / np.sqrt(len(cortical_data))]}),
            pd.DataFrame({'Category': ['Subcortical'], 'mean_pct_normalized': [subcortical_data['count_pct_brain'].sum()], 
                         'sem_pct_normalized': [subcortical_data['count_pct_brain'].std() / np.sqrt(len(subcortical_data))]})
        ])
        cortical_vs_subcortical.to_excel(writer, sheet_name='1_Cortical_vs_Subcortical', index=False)
        
        # Sheet 3: Subcortical regions
        subcortical_regions = calculate_group_stats(details_df[subcortical_mask], 'allen_primary_group')
        subcortical_regions.to_excel(writer, sheet_name='2_Subcortical_Regions', index=False)
        
        # Sheet 4: Cortical regions
        cortical_regions = calculate_group_stats(details_df[cortical_mask], 'allen_cortical_class')
        cortical_regions.to_excel(writer, sheet_name='3_Cortical_Regions', index=False)
        
        # Sheet 5: Thalamus DORpm and DORsm
        thalamus_data = details_df[details_df['allen_primary_group'] == 'Thalamus']
        thalamus_groups = calculate_group_stats(thalamus_data, 'thalamic_group')
        thalamus_groups.to_excel(writer, sheet_name='4_Thalamus_DORpm_DORsm', index=False)
        
        # Sheet 6: DORpm subdivisions
        dorpm_data = thalamus_data[thalamus_data['thalamic_group'] == 'Thalamus, polymodal association cortex related']
        dorpm_subdivisions = calculate_group_stats(dorpm_data, 'thalamic_subdivision')
        dorpm_subdivisions.to_excel(writer, sheet_name='5_DORpm_Subdivisions', index=False)
        
        # Sheet 7: DORsm subdivisions  
        dorsm_data = thalamus_data[thalamus_data['thalamic_group'] == 'Thalamus, sensory-motor cortex related']
        dorsm_subdivisions = calculate_group_stats(dorsm_data, 'thalamic_subdivision')
        dorsm_subdivisions.to_excel(writer, sheet_name='6_DORsm_Subdivisions', index=False)
        
        # Sheet 8: Intralaminar nuclei
        intralaminar_data = thalamus_data[thalamus_data['thalamic_subdivision'] == 'Intralaminar nuclei of the dorsal thalamus']
        intralaminar_stats = calculate_group_stats(intralaminar_data, 'Region')
        intralaminar_stats.to_excel(writer, sheet_name='7_Intralaminar_Nuclei', index=False)
        
        # Sheet 9: Isocortex subdivisions
        isocortex_data = details_df[details_df['allen_cortical_class'] == 'Isocortex']
        isocortex_stats = calculate_group_stats(isocortex_data, 'allen_parent')
        isocortex_stats.to_excel(writer, sheet_name='8_Isocortex_Subdivisions', index=False)
        
        # Sheet 10: Basal ganglia
        basal_ganglia_data = details_df[details_df['allen_primary_group'].isin(['Striatum', 'Pallidum'])]
        basal_ganglia_stats = calculate_group_stats(basal_ganglia_data, 'allen_primary_group')
        basal_ganglia_stats.to_excel(writer, sheet_name='9_Basal_Ganglia', index=False)
        
        # Sheet 11: Cortex regions per brain (for heatmap)
        cortex_regions_data = details_df[details_df['is_cortical'] == True].groupby(['Region', 'source_sheet'])['count_pct_brain'].sum().reset_index()
        cortex_regions_pivot = cortex_regions_data.pivot(index='Region', columns='source_sheet', values='count_pct_brain').fillna(0)
        # Normalize each brain to 100% within cortex
        cortex_regions_pivot = cortex_regions_pivot.div(cortex_regions_pivot.sum(axis=0), axis=1) * 100
        cortex_regions_pivot.to_excel(writer, sheet_name='10_Cortex_Regions_Heatmap_Data', index=True)
        
        # Sheet 12: Subcortex regions per brain (for heatmap)
        subcortex_regions_data = details_df[details_df['is_subcortical'] == True].groupby(['Region', 'source_sheet'])['count_pct_brain'].sum().reset_index()
        subcortex_regions_pivot = subcortex_regions_data.pivot(index='Region', columns='source_sheet', values='count_pct_brain').fillna(0)
        # Normalize each brain to 100% within subcortex
        subcortex_regions_pivot = subcortex_regions_pivot.div(subcortex_regions_pivot.sum(axis=0), axis=1) * 100
        subcortex_regions_pivot.to_excel(writer, sheet_name='11_Subcortex_Regions_Heatmap_Data', index=True)
    
    print(f"Excel file saved: {excel_path}")

def main():
    """Main analysis function - Now divides subcortex and cortex for each region"""
    print("Loading data...")
    details_df, summary_df = load_data()
    
    print("Filtering data for minimum cell threshold (2 cells)...")
    details_df = filter_minimum_cells(details_df)
    
    print("Creating graphs with cortical/subcortical division for each region...")
    
    # Graph 1: Cortical vs Subcortical
    print("Creating Graph 1: Cortical vs Subcortical...")
    cortical_mask = details_df['is_cortical'] == True
    subcortical_mask = details_df['is_subcortical'] == True
    
    cortical_data = details_df[cortical_mask]
    subcortical_data = details_df[subcortical_mask]
    
    cortical_vs_subcortical = pd.DataFrame({
        'Category': ['Cortical', 'Subcortical'],
        'mean_pct_normalized': [
            cortical_data['count_pct_brain'].sum(),
            subcortical_data['count_pct_brain'].sum()
        ],
        'sem_pct_normalized': [
            cortical_data['count_pct_brain'].std() / np.sqrt(len(cortical_data)),
            subcortical_data['count_pct_brain'].std() / np.sqrt(len(subcortical_data))
        ]
    })
    
    # Normalize to 100%
    total = cortical_vs_subcortical['mean_pct_normalized'].sum()
    cortical_vs_subcortical['mean_pct_normalized'] = (cortical_vs_subcortical['mean_pct_normalized'] / total) * 100
    cortical_vs_subcortical['sem_pct_normalized'] = (cortical_vs_subcortical['sem_pct_normalized'] / total) * 100
    
    create_bar_plot(cortical_vs_subcortical, 
                   "Cortical vs Subcortical Regions (Normalized to 100%)",
                   FIGURES_DIR / "1_cortical_vs_subcortical")
    
    # Graph 2: Subcortical regions (normalized within subcortex)
    print("Creating Graph 2: Subcortical regions...")
    subcortical_regions = calculate_group_stats(details_df[subcortical_mask], 'allen_primary_group')
    create_bar_plot(subcortical_regions,
                   "Subcortical Regions (Normalized to 100% within Subcortex)",
                   FIGURES_DIR / "2_subcortical_regions")
    
    # Graph 3: Cortical regions (normalized within cortex)
    print("Creating Graph 3: Cortical regions...")
    cortical_regions = calculate_group_stats(details_df[cortical_mask], 'allen_cortical_class')
    create_bar_plot(cortical_regions,
                   "Cortical Regions (Normalized to 100% within Cortex)",
                   FIGURES_DIR / "3_cortical_regions")
    
    # Graph 4: Thalamus DORpm and DORsm (normalized within thalamus)
    print("Creating Graph 4: Thalamus DORpm and DORsm...")
    thalamus_data = details_df[details_df['allen_primary_group'] == 'Thalamus']
    thalamus_groups = calculate_group_stats(thalamus_data, 'thalamic_group')
    create_bar_plot(thalamus_groups,
                   "Thalamus: DORpm vs DORsm (Normalized to 100% within Thalamus)",
                   FIGURES_DIR / "4_thalamus_dorpm_dorsm")
    
    # Graph 5: DORpm subdivisions (normalized within DORpm)
    print("Creating Graph 5: DORpm subdivisions...")
    dorpm_data = thalamus_data[thalamus_data['thalamic_group'] == 'Thalamus, polymodal association cortex related']
    dorpm_subdivisions = calculate_group_stats(dorpm_data, 'thalamic_subdivision')
    create_bar_plot(dorpm_subdivisions,
                   "DORpm Subdivisions (Normalized to 100% within DORpm)",
                   FIGURES_DIR / "5_dorpm_subdivisions")
    
    # Graph 6: DORsm subdivisions (normalized within DORsm)
    print("Creating Graph 6: DORsm subdivisions...")
    dorsm_data = thalamus_data[thalamus_data['thalamic_group'] == 'Thalamus, sensory-motor cortex related']
    dorsm_subdivisions = calculate_group_stats(dorsm_data, 'thalamic_subdivision')
    create_bar_plot(dorsm_subdivisions,
                   "DORsm Subdivisions (Normalized to 100% within DORsm)",
                   FIGURES_DIR / "6_dorsm_subdivisions")
    
    # Graph 7: Intralaminar nuclei (normalized within intralaminar)
    print("Creating Graph 7: Intralaminar nuclei...")
    intralaminar_data = thalamus_data[thalamus_data['thalamic_subdivision'] == 'Intralaminar nuclei of the dorsal thalamus']
    intralaminar_stats = calculate_group_stats(intralaminar_data, 'Region')
    create_bar_plot(intralaminar_stats,
                   "Intralaminar Nuclei (Normalized to 100% within Intralaminar)",
                   FIGURES_DIR / "7_intralaminar_nuclei")
    
    # Graph 8: Isocortex subdivisions (normalized within isocortex)
    print("Creating Graph 8: Isocortex subdivisions...")
    isocortex_data = details_df[details_df['allen_cortical_class'] == 'Isocortex']
    isocortex_stats = calculate_group_stats(isocortex_data, 'allen_parent')
    create_bar_plot(isocortex_stats,
                   "Isocortex Subdivisions (Normalized to 100% within Isocortex)",
                   FIGURES_DIR / "8_isocortex_subdivisions")
    
    # Graph 9: Basal ganglia (normalized within basal ganglia)
    print("Creating Graph 9: Basal ganglia...")
    basal_ganglia_data = details_df[details_df['allen_primary_group'].isin(['Striatum', 'Pallidum'])]
    basal_ganglia_stats = calculate_group_stats(basal_ganglia_data, 'allen_primary_group')
    create_bar_plot(basal_ganglia_stats,
                   "Basal Ganglia (Normalized to 100% within Basal Ganglia)",
                   FIGURES_DIR / "9_basal_ganglia")
    
    # Graph 10: Heat map - Cortex regions only (normalized per brain within cortex)
    print("Creating Graph 10: Heat map - Cortex regions only...")
    cortex_regions_data = details_df[cortical_mask].groupby(['Region', 'source_sheet'])['count_pct_brain'].sum().reset_index()
    create_heatmap(cortex_regions_data,
                  "Cortex Regions per Brain (Normalized to 100% per Brain within Cortex)",
                  FIGURES_DIR / "10_heatmap_cortex_regions",
                  figsize=(15, 20))
    
    # Graph 11: Heat map - Subcortex regions only (normalized per brain within subcortex)
    print("Creating Graph 11: Heat map - Subcortex regions only...")
    subcortex_regions_data = details_df[subcortical_mask].groupby(['Region', 'source_sheet'])['count_pct_brain'].sum().reset_index()
    create_heatmap(subcortex_regions_data,
                  "Subcortex Regions per Brain (Normalized to 100% per Brain within Subcortex)",
                  FIGURES_DIR / "11_heatmap_subcortex_regions",
                  figsize=(15, 20))
    
    # Create Excel file
    print("Creating Excel file...")
    create_excel_sheets(details_df, summary_df)
    
    print(f"\nAnalysis complete! Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Excel files saved to: {EXCEL_DIR}")
    print("\nAll figures saved as PNG, SVG, and PDF for Illustrator editing!")

if __name__ == "__main__":
    main()
