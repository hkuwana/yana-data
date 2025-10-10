#!/usr/bin/env python3
"""
Enhanced Brain Analysis Script

This script creates enhanced visualizations with:
1. Heatmaps of parent groups for subcortical and cortical regions
2. Individual brain values displayed on graphs
3. Comprehensive output tables with normalized values per brain
4. Detailed README documentation

The script addresses the user's request for:
- Heatmaps of parent groups (subcortical and cortical)
- Individual brain values per brain added to graphs
- Output tables with normalization showing work per brain
- README documentation
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
RESULTS_DIR = DATA_DIR / "enhanced_results"
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

def filter_excluded_regions(df):
    """Filter out basic cell groups, claustrum, cerebral cortex, endopiriform nucleus, and 'Other' allen_primary_group regions (except amygdalar)"""
    excluded_regions = [
        'Basic cell groups and regions',
        'Claustrum', 
        'Cerebral cortex',
        'Endopiriform nucleus'
    ]
    
    # Filter out excluded regions
    filtered_df = df[~df['Region'].isin(excluded_regions)].copy()
    
    # Filter out 'Other' allen_primary_group regions, but keep amygdalar regions
    amygdalar_mask = filtered_df['Region'].str.contains('amygdalar', case=False, na=False)
    specific_amygdalar_regions = [
        'Cortical amygdalar area',
        'Piriform-amygdalar area'
    ]
    specific_amygdalar_mask = filtered_df['Region'].isin(specific_amygdalar_regions)
    all_amygdalar_mask = amygdalar_mask | specific_amygdalar_mask
    
    # Keep amygdalar regions even if they have 'Other' allen_primary_group
    other_mask = filtered_df['allen_primary_group'] == 'Other'
    other_non_amygdalar_mask = other_mask & ~all_amygdalar_mask
    
    # Get regions that will be excluded due to 'Other' allen_primary_group
    other_excluded_regions = filtered_df[other_non_amygdalar_mask]['Region'].unique()
    
    # Apply the filter
    filtered_df = filtered_df[~other_non_amygdalar_mask].copy()
    
    # Log excluded regions
    excluded_count = len(df) - len(filtered_df)
    if excluded_count > 0:
        excluded_log = RESULTS_DIR / "excluded_regions_filter.txt"
        with open(excluded_log, 'w') as f:
            f.write("Regions excluded by user request:\n")
            f.write("=" * 50 + "\n")
            
            # Log explicitly excluded regions
            for region in excluded_regions:
                if region in df['Region'].values:
                    count = df[df['Region'] == region]['count'].sum()
                    f.write(f"{region}: {count} cells\n")
            
            # Log 'Other' allen_primary_group regions (non-amygdalar)
            if len(other_excluded_regions) > 0:
                f.write(f"\nRegions excluded due to 'Other' allen_primary_group (non-amygdalar):\n")
                f.write("-" * 50 + "\n")
                for region in other_excluded_regions:
                    count = df[df['Region'] == region]['count'].sum()
                    f.write(f"{region}: {count} cells\n")
            
            # Add footnote about amygdalar regions
            f.write(f"\nFootnote: Amygdalar regions with 'Other' allen_primary_group were kept and grouped under 'Amygdala'.\n")
        
        print(f"Excluded {excluded_count} regions by user request. See {excluded_log}")
    
    return filtered_df

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

def create_enhanced_bar_plot(stats, title, output_path, individual_data=None, figsize=(12, 8)):
    """Create an enhanced bar plot with individual brain values"""
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
    plt.xlabel("Mean percentage (%)")
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(stats['mean_pct_normalized'], stats['sem_pct_normalized'])):
        plt.text(mean + sem + 1, i, f'{mean:.1f}±{sem:.1f}', 
                va='center', fontsize=8)
    
    # Add individual brain values if provided
    if individual_data is not None:
        # Create a second y-axis for individual brain values
        ax2 = plt.gca().twinx()
        ax2.set_ylim(plt.gca().get_ylim())
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])  # Hide y-axis labels for the second axis
        
        # Add individual brain values as text
        for i, group in enumerate(stats.iloc[:, 0]):
            if group in individual_data.columns:
                brain_values = individual_data[group].dropna()
                if len(brain_values) > 0:
                    # Position text to the right of the bars
                    max_val = stats['mean_pct_normalized'].iloc[i] + stats['sem_pct_normalized'].iloc[i]
                    text_x = max_val + 5
                    text_y = i
                    
                    # Create text with individual values
                    individual_text = f"Individual brains:\n" + "\n".join([f"Brain{j+1}: {val:.1f}%" 
                                                                        for j, val in enumerate(brain_values)])
                    ax2.text(text_x, text_y, individual_text, va='center', fontsize=6, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save as PNG, SVG, and PDF
    base_path = output_path.with_suffix('')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.svg", bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {base_path}.png, .svg, .pdf")

def create_parent_group_heatmap(data, title, output_path, parent_col, figsize=(12, 8)):
    """Create a heatmap for parent groups with individual brain values"""
    plt.figure(figsize=figsize)
    
    # Group by parent and brain, then normalize within each brain
    grouped_data = data.groupby([parent_col, 'source_sheet'])['count_pct_brain'].sum().reset_index()
    pivot_data = grouped_data.pivot_table(
        values='count_pct_brain',
        index=parent_col,
        columns='source_sheet',
        fill_value=0
    )
    
    # Normalize each brain to 100% within the category
    pivot_data = pivot_data.div(pivot_data.sum(axis=0), axis=1) * 100
    
    # Create heatmap with better formatting
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', 
                cbar_kws={'label': 'Percentage (%)'}, 
                annot_kws={'size': 8})
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Brain', fontsize=12)
    plt.ylabel('Parent Group', fontsize=12)
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
    
    return pivot_data

def calculate_group_stats_with_individuals(df, group_col, value_col='count_pct_brain', min_brains=3):
    """Calculate mean and SEM for grouped data, plus individual brain values"""
    # Filter for regions present in minimum number of brains
    brain_counts = df.groupby(group_col)['source_sheet'].nunique()
    valid_groups = brain_counts[brain_counts >= min_brains].index
    filtered_df = df[df[group_col].isin(valid_groups)].copy()
    
    if filtered_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
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
    
    # Create individual brain values table
    individual_data = filtered_df.pivot_table(
        values=value_col,
        index=group_col,
        columns='source_sheet',
        fill_value=np.nan
    )
    
    # Normalize individual values within each brain
    individual_data = individual_data.div(individual_data.sum(axis=0), axis=1) * 100
    
    return stats.sort_values('mean_pct_normalized', ascending=False), individual_data

def create_comprehensive_excel(details_df, summary_df, subcortical_heatmap_data=None, cortical_heatmap_data=None, 
                              subcortical_stats=None, cortical_stats=None, subcortical_individual=None, cortical_individual=None):
    """Create comprehensive Excel file with all data and individual brain values for every graph"""
    excel_path = EXCEL_DIR / "enhanced_brain_analysis.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Raw data summary
        summary_df.to_excel(writer, sheet_name='Raw_Data_Summary', index=False)
        
        # Sheet 2: Complete raw data for reproducibility
        details_df.to_excel(writer, sheet_name='Complete_Raw_Data', index=False)
        
        # Sheets 3-6: Heatmap data (the data used to create heatmaps)
        if subcortical_heatmap_data is not None:
            subcortical_heatmap_data.to_excel(writer, sheet_name='Subcortical_Heatmap_Data', index=True)
        
        if cortical_heatmap_data is not None:
            cortical_heatmap_data.to_excel(writer, sheet_name='Cortical_Heatmap_Data', index=True)
        
        # Sheets 7-10: Bar plot data (summary statistics and individual values)
        if subcortical_stats is not None:
            subcortical_stats.to_excel(writer, sheet_name='Subcortical_BarPlot_Stats', index=False)
        
        if cortical_stats is not None:
            cortical_stats.to_excel(writer, sheet_name='Cortical_BarPlot_Stats', index=False)
        
        if subcortical_individual is not None:
            subcortical_individual.to_excel(writer, sheet_name='Subcortical_Individual_Brains', index=True)
        
        if cortical_individual is not None:
            cortical_individual.to_excel(writer, sheet_name='Cortical_Individual_Brains', index=True)
        
        # Sheets 11-14: Raw data used for each graph (for complete reproducibility)
        # Subcortical data used in graphs
        subcortical_mask = details_df['is_subcortical'] == True
        amygdalar_mask = details_df['Region'].str.contains('amygdalar', case=False, na=False)
        specific_amygdalar_regions = [
            'Cortical amygdalar area',
            'Piriform-amygdalar area'
        ]
        specific_amygdalar_mask = details_df['Region'].isin(specific_amygdalar_regions)
        all_amygdalar_mask = amygdalar_mask | specific_amygdalar_mask
        subcortical_mask = subcortical_mask | all_amygdalar_mask
        subcortical_data = details_df[subcortical_mask]
        subcortical_data.to_excel(writer, sheet_name='Subcortical_Raw_Data', index=False)
        
        # Cortical data used in graphs
        cortical_mask = details_df['is_cortical'] == True
        cortical_mask = cortical_mask & ~all_amygdalar_mask  # Exclude amygdalar regions from cortical
        cortical_data = details_df[cortical_mask]
        cortical_data.to_excel(writer, sheet_name='Cortical_Raw_Data', index=False)
        
        # Sheet 15: All regions individual values (for reference)
        all_regions_individual = details_df.groupby(['Region', 'source_sheet'])['count_pct_brain'].sum().reset_index()
        all_regions_pivot = all_regions_individual.pivot(index='Region', columns='source_sheet', values='count_pct_brain').fillna(0)
        all_regions_pivot.to_excel(writer, sheet_name='All_Regions_Individual', index=True)
        
        # Sheet 16: Graph metadata and instructions
        graph_info = pd.DataFrame({
            'Graph_Name': [
                'Subcortical_Heatmap',
                'Cortical_Heatmap', 
                'Subcortical_BarPlot',
                'Cortical_BarPlot'
            ],
            'Data_Sheet': [
                'Subcortical_Heatmap_Data',
                'Cortical_Heatmap_Data',
                'Subcortical_BarPlot_Stats + Subcortical_Individual_Brains',
                'Cortical_BarPlot_Stats + Cortical_Individual_Brains'
            ],
            'Raw_Data_Sheet': [
                'Subcortical_Raw_Data',
                'Cortical_Raw_Data',
                'Subcortical_Raw_Data',
                'Cortical_Raw_Data'
            ],
            'Description': [
                'Heatmap showing custom subcortical groups normalized to 100% per brain',
                'Heatmap showing custom cortical groups normalized to 100% per brain',
                'Bar plot with mean±SEM and individual brain values for subcortical groups',
                'Bar plot with mean±SEM and individual brain values for cortical groups'
            ]
        })
        graph_info.to_excel(writer, sheet_name='Graph_Data_Guide', index=False)
    
    print(f"Excel file saved: {excel_path}")
    print(f"Excel contains {len(pd.ExcelFile(excel_path).sheet_names)} sheets with all graph data")

def create_custom_cortical_groups(details_df):
    """Create custom cortical groupings based on functional areas"""
    df = details_df.copy()
    
    # Create custom cortical groups - initialize with Allen cortical class as base
    df['custom_cortical_group'] = df['allen_cortical_class'].copy()
    
    # Somatosensory regions (in isocortex) -> Somatosensory Cortex
    somatosensory_regions = [
        'Primary somatosensory area',
        'Supplemental somatosensory area',
        'Visceral area'
    ]
    somatosensory_mask = df['Region'].isin(somatosensory_regions)
    df.loc[somatosensory_mask, 'custom_cortical_group'] = 'Somatosensory Cortex'
    
    # Motor areas -> Somatomotor Area
    motor_regions = [
        'Primary motor area',
        'Secondary motor area'
    ]
    motor_mask = df['Region'].isin(motor_regions)
    df.loc[motor_mask, 'custom_cortical_group'] = 'Somatomotor Area'
    
    # Gustatory area -> Gustatory Area
    gustatory_mask = df['Region'].str.contains('gustatory', case=False, na=False)
    df.loc[gustatory_mask, 'custom_cortical_group'] = 'Gustatory Area'
    
    # Visual areas -> Visual Areas
    visual_regions = [
        'Primary visual area',
        'Anterolateral visual area',
        'Anteromedial visual area'
    ]
    visual_mask = df['Region'].isin(visual_regions)
    df.loc[visual_mask, 'custom_cortical_group'] = 'Visual Areas'
    
    # Medial prefrontal cortex regions -> Medial Prefrontal Cortex
    mpfc_regions = [
        'Anterior cingulate area',
        'Infralimbic area',
        'Prelimbic area',
        'Orbital area'
    ]
    mpfc_mask = df['Region'].isin(mpfc_regions)
    df.loc[mpfc_mask, 'custom_cortical_group'] = 'Medial Prefrontal Cortex'
    
    # Auditory areas -> Auditory Areas
    auditory_regions = [
        'Primary auditory area',
        'Dorsal auditory area',
        'Ventral auditory area'
    ]
    auditory_mask = df['Region'].isin(auditory_regions)
    df.loc[auditory_mask, 'custom_cortical_group'] = 'Auditory Areas'
    
    # Association areas -> Association Areas
    association_regions = [
        'Posterior parietal association areas',
        'Temporal association areas',
        'Retrosplenial area'
    ]
    association_mask = df['Region'].isin(association_regions)
    df.loc[association_mask, 'custom_cortical_group'] = 'Association Areas'
    
    # Insular areas -> Insular Areas
    insular_regions = [
        'Agranular insular area'
    ]
    insular_mask = df['Region'].isin(insular_regions)
    df.loc[insular_mask, 'custom_cortical_group'] = 'Insular Areas'
    
    # Perirhinal/Ectorhinal areas -> Perirhinal Areas
    perirhinal_regions = [
        'Perirhinal area',
        'Ectorhinal area/Layer 1',
        'Ectorhinal area/Layer 2/3',
        'Ectorhinal area/Layer 5',
        'Ectorhinal area/Layer 6a'
    ]
    perirhinal_mask = df['Region'].isin(perirhinal_regions)
    df.loc[perirhinal_mask, 'custom_cortical_group'] = 'Perirhinal Areas'
    
    return df

def create_custom_subcortical_groups(details_df):
    """Create custom subcortical groupings including amygdala"""
    df = details_df.copy()
    
    # Create custom subcortical groups
    df['custom_subcortical_group'] = df['allen_primary_group'].copy()
    
    # Group ALL amygdalar regions into Amygdala (regardless of original classification)
    amygdalar_mask = df['Region'].str.contains('amygdalar', case=False, na=False)
    df.loc[amygdalar_mask, 'custom_subcortical_group'] = 'Amygdala'
    
    # Also ensure cortical amygdalar area and piriform-amygdalar area are included
    specific_amygdalar_regions = [
        'Cortical amygdalar area',
        'Piriform-amygdalar area'
    ]
    specific_mask = df['Region'].isin(specific_amygdalar_regions)
    df.loc[specific_mask, 'custom_subcortical_group'] = 'Amygdala'
    
    return df

def print_custom_cortical_mapping_info(details_df):
    """Print detailed custom cortical grouping information"""
    print("\n" + "="*80)
    print("CUSTOM CORTICAL GROUPING INFORMATION")
    print("="*80)
    
    # Apply custom groupings
    df_with_custom = create_custom_cortical_groups(details_df)
    
    # Filter out amygdalar regions from cortical data
    cortical_mask = df_with_custom['is_cortical'] == True
    amygdalar_mask = df_with_custom['Region'].str.contains('amygdalar', case=False, na=False)
    specific_amygdalar_regions = [
        'Cortical amygdalar area',
        'Piriform-amygdalar area'
    ]
    specific_amygdalar_mask = df_with_custom['Region'].isin(specific_amygdalar_regions)
    all_amygdalar_mask = amygdalar_mask | specific_amygdalar_mask
    cortical_mask = cortical_mask & ~all_amygdalar_mask
    cortical_data = df_with_custom[cortical_mask]
    custom_groups = cortical_data.groupby('custom_cortical_group')
    
    print("\nCUSTOM CORTICAL GROUPS:")
    print("-" * 50)
    
    for group_name, group_data in custom_groups:
        print(f"\n{group_name.upper()}:")
        regions = group_data['Region'].unique()
        for region in sorted(regions):
            region_info = group_data[group_data['Region'] == region].iloc[0]
            allen_id = region_info['allen_id']
            allen_acronym = region_info['allen_acronym']
            allen_name = region_info['allen_name']
            allen_cortical_class = region_info['allen_cortical_class']
            print(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})")
            print(f"    Allen Cortical Class: {allen_cortical_class}")
            if allen_name != region:
                print(f"    Allen Name: {allen_name}")
    
    print("\n" + "="*80)
    print("END OF CUSTOM CORTICAL GROUPING INFORMATION")
    print("="*80 + "\n")

def print_allen_mapping_info(details_df):
    """Print detailed Allen Brain Atlas mapping information"""
    print("\n" + "="*80)
    print("ALLEN BRAIN ATLAS MAPPING INFORMATION")
    print("="*80)
    
    # Subcortical regions mapping
    print("\nSUBCORTICAL REGIONS MAPPING:")
    print("-" * 50)
    subcortical_data = details_df[details_df['is_subcortical'] == True]
    subcortical_groups = subcortical_data.groupby('allen_primary_group')
    
    for group_name, group_data in subcortical_groups:
        print(f"\n{group_name.upper()}:")
        regions = group_data['Region'].unique()
        for region in sorted(regions):
            # Get Allen ID and acronym for this region
            region_info = group_data[group_data['Region'] == region].iloc[0]
            allen_id = region_info['allen_id']
            allen_acronym = region_info['allen_acronym']
            allen_name = region_info['allen_name']
            print(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})")
            if allen_name != region:
                print(f"    Allen Name: {allen_name}")
    
    # Cortical regions mapping
    print("\n\nCORTICAL REGIONS MAPPING:")
    print("-" * 50)
    cortical_data = details_df[details_df['is_cortical'] == True]
    cortical_groups = cortical_data.groupby('allen_cortical_class')
    
    for group_name, group_data in cortical_groups:
        print(f"\n{group_name.upper()}:")
        regions = group_data['Region'].unique()
        for region in sorted(regions):
            # Get Allen ID and acronym for this region
            region_info = group_data[group_data['Region'] == region].iloc[0]
            allen_id = region_info['allen_id']
            allen_acronym = region_info['allen_acronym']
            allen_name = region_info['allen_name']
            print(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})")
            if allen_name != region:
                print(f"    Allen Name: {allen_name}")
    
    # Thalamic subdivisions mapping
    print("\n\nTHALAMIC SUBDIVISIONS MAPPING:")
    print("-" * 50)
    thalamus_data = details_df[details_df['allen_primary_group'] == 'Thalamus']
    thalamic_groups = thalamus_data.groupby('thalamic_group')
    
    for group_name, group_data in thalamic_groups:
        print(f"\n{group_name.upper()}:")
        subdivisions = group_data.groupby('thalamic_subdivision')
        for subdiv_name, subdiv_data in subdivisions:
            print(f"  {subdiv_name}:")
            regions = subdiv_data['Region'].unique()
            for region in sorted(regions):
                region_info = subdiv_data[subdiv_data['Region'] == region].iloc[0]
                allen_id = region_info['allen_id']
                allen_acronym = region_info['allen_acronym']
                allen_name = region_info['allen_name']
                print(f"    • {region} (ID: {allen_id}, Acronym: {allen_acronym})")
                if allen_name != region:
                    print(f"      Allen Name: {allen_name}")
    
    print("\n" + "="*80)
    print("END OF ALLEN BRAIN ATLAS MAPPING INFORMATION")
    print("="*80 + "\n")

def create_allen_mapping_report(details_df):
    """Create a detailed Allen Brain Atlas mapping report file with custom groupings"""
    report_path = RESULTS_DIR / "allen_brain_atlas_mapping_report.txt"
    
    # Apply custom cortical groupings
    df_with_custom = create_custom_cortical_groups(details_df)
    
    with open(report_path, 'w') as f:
        f.write("ALLEN BRAIN ATLAS MAPPING REPORT WITH CUSTOM CORTICAL GROUPINGS\n")
        f.write("=" * 80 + "\n")
        f.write("This report shows how each brain region is mapped to Allen Brain Atlas groups\n")
        f.write("and custom cortical functional groupings.\n")
        f.write("Generated by enhanced_brain_analysis.py\n\n")
        
        # Subcortical regions mapping
        f.write("SUBCORTICAL REGIONS MAPPING:\n")
        f.write("-" * 50 + "\n")
        subcortical_data = details_df[details_df['is_subcortical'] == True]
        subcortical_groups = subcortical_data.groupby('allen_primary_group')
        
        for group_name, group_data in subcortical_groups:
            f.write(f"\n{group_name.upper()}:\n")
            regions = group_data['Region'].unique()
            for region in sorted(regions):
                region_info = group_data[group_data['Region'] == region].iloc[0]
                allen_id = region_info['allen_id']
                allen_acronym = region_info['allen_acronym']
                allen_name = region_info['allen_name']
                allen_parent = region_info['allen_parent']
                f.write(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})\n")
                f.write(f"    Allen Name: {allen_name}\n")
                f.write(f"    Allen Parent: {allen_parent}\n")
                f.write(f"    Total Cells: {group_data[group_data['Region'] == region]['count'].sum()}\n")
                f.write(f"    Present in Brains: {group_data[group_data['Region'] == region]['source_sheet'].nunique()}\n\n")
        
        # Cortical regions mapping
        f.write("\n\nCORTICAL REGIONS MAPPING:\n")
        f.write("-" * 50 + "\n")
        cortical_data = details_df[details_df['is_cortical'] == True]
        cortical_groups = cortical_data.groupby('allen_cortical_class')
        
        for group_name, group_data in cortical_groups:
            f.write(f"\n{group_name.upper()}:\n")
            regions = group_data['Region'].unique()
            for region in sorted(regions):
                region_info = group_data[group_data['Region'] == region].iloc[0]
                allen_id = region_info['allen_id']
                allen_acronym = region_info['allen_acronym']
                allen_name = region_info['allen_name']
                allen_parent = region_info['allen_parent']
                f.write(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})\n")
                f.write(f"    Allen Name: {allen_name}\n")
                f.write(f"    Allen Parent: {allen_parent}\n")
                f.write(f"    Total Cells: {group_data[group_data['Region'] == region]['count'].sum()}\n")
                f.write(f"    Present in Brains: {group_data[group_data['Region'] == region]['source_sheet'].nunique()}\n\n")
        
        # Thalamic subdivisions mapping
        f.write("\n\nTHALAMIC SUBDIVISIONS MAPPING:\n")
        f.write("-" * 50 + "\n")
        thalamus_data = details_df[details_df['allen_primary_group'] == 'Thalamus']
        thalamic_groups = thalamus_data.groupby('thalamic_group')
        
        for group_name, group_data in thalamic_groups:
            f.write(f"\n{group_name.upper()}:\n")
            subdivisions = group_data.groupby('thalamic_subdivision')
            for subdiv_name, subdiv_data in subdivisions:
                f.write(f"  {subdiv_name}:\n")
                regions = subdiv_data['Region'].unique()
                for region in sorted(regions):
                    region_info = subdiv_data[subdiv_data['Region'] == region].iloc[0]
                    allen_id = region_info['allen_id']
                    allen_acronym = region_info['allen_acronym']
                    allen_name = region_info['allen_name']
                    allen_parent = region_info['allen_parent']
                    f.write(f"    • {region} (ID: {allen_id}, Acronym: {allen_acronym})\n")
                    f.write(f"      Allen Name: {allen_name}\n")
                    f.write(f"      Allen Parent: {allen_parent}\n")
                    f.write(f"      Total Cells: {subdiv_data[subdiv_data['Region'] == region]['count'].sum()}\n")
                    f.write(f"      Present in Brains: {subdiv_data[subdiv_data['Region'] == region]['source_sheet'].nunique()}\n\n")
        
        # Custom subcortical groupings
        f.write("\n\nCUSTOM SUBCORTICAL FUNCTIONAL GROUPINGS:\n")
        f.write("-" * 50 + "\n")
        subcortical_mask = df_with_custom['is_subcortical'] == True
        amygdalar_mask = df_with_custom['Region'].str.contains('amygdalar', case=False, na=False)
        specific_amygdalar_regions = [
            'Cortical amygdalar area',
            'Piriform-amygdalar area'
        ]
        specific_amygdalar_mask = df_with_custom['Region'].isin(specific_amygdalar_regions)
        all_amygdalar_mask = amygdalar_mask | specific_amygdalar_mask
        subcortical_mask = subcortical_mask | all_amygdalar_mask
        subcortical_data_custom = df_with_custom[subcortical_mask]
        custom_subcortical_groups = subcortical_data_custom.groupby('custom_subcortical_group')
        
        for group_name, group_data in custom_subcortical_groups:
            f.write(f"\n{group_name.upper()}:\n")
            regions = group_data['Region'].unique()
            for region in sorted(regions):
                region_info = group_data[group_data['Region'] == region].iloc[0]
                allen_id = region_info['allen_id']
                allen_acronym = region_info['allen_acronym']
                allen_name = region_info['allen_name']
                allen_primary_group = region_info['allen_primary_group']
                f.write(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})\n")
                f.write(f"    Allen Name: {allen_name}\n")
                f.write(f"    Allen Primary Group: {allen_primary_group}\n")
                f.write(f"    Total Cells: {group_data[group_data['Region'] == region]['count'].sum()}\n")
                f.write(f"    Present in Brains: {group_data[group_data['Region'] == region]['source_sheet'].nunique()}\n\n")
        
        # Custom cortical groupings
        f.write("\n\nCUSTOM CORTICAL FUNCTIONAL GROUPINGS:\n")
        f.write("-" * 50 + "\n")
        cortical_data_custom = df_with_custom[df_with_custom['is_cortical'] == True]
        # Exclude amygdalar regions from cortical custom groups
        amygdalar_mask_for_cortical = df_with_custom['Region'].str.contains('amygdalar', case=False, na=False)
        specific_amygdalar_regions_for_cortical = [
            'Cortical amygdalar area',
            'Piriform-amygdalar area'
        ]
        specific_amygdalar_mask_for_cortical = df_with_custom['Region'].isin(specific_amygdalar_regions_for_cortical)
        all_amygdalar_mask_for_cortical = amygdalar_mask_for_cortical | specific_amygdalar_mask_for_cortical
        cortical_data_custom = cortical_data_custom[~all_amygdalar_mask_for_cortical]
        custom_groups = cortical_data_custom.groupby('custom_cortical_group')
        
        for group_name, group_data in custom_groups:
            f.write(f"\n{group_name.upper()}:\n")
            regions = group_data['Region'].unique()
            for region in sorted(regions):
                region_info = group_data[group_data['Region'] == region].iloc[0]
                allen_id = region_info['allen_id']
                allen_acronym = region_info['allen_acronym']
                allen_name = region_info['allen_name']
                allen_cortical_class = region_info['allen_cortical_class']
                allen_parent = region_info['allen_parent']
                f.write(f"  • {region} (ID: {allen_id}, Acronym: {allen_acronym})\n")
                f.write(f"    Allen Name: {allen_name}\n")
                f.write(f"    Allen Cortical Class: {allen_cortical_class}\n")
                f.write(f"    Allen Parent: {allen_parent}\n")
                f.write(f"    Total Cells: {group_data[group_data['Region'] == region]['count'].sum()}\n")
                f.write(f"    Present in Brains: {group_data[group_data['Region'] == region]['source_sheet'].nunique()}\n\n")
        
        # Summary statistics
        f.write("\n\nSUMMARY STATISTICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total regions analyzed: {len(details_df['Region'].unique())}\n")
        f.write(f"Subcortical regions: {len(subcortical_data['Region'].unique())}\n")
        f.write(f"Cortical regions: {len(cortical_data['Region'].unique())}\n")
        f.write(f"Thalamic regions: {len(thalamus_data['Region'].unique())}\n")
        f.write(f"Custom cortical groups: {len(custom_groups)}\n")
        f.write(f"Total cells across all regions: {details_df['count'].sum()}\n")
        f.write(f"Number of brains: {details_df['source_sheet'].nunique()}\n")
        
        # Excluded regions
        excluded_regions = details_df[~details_df['Region'].isin(details_df.groupby('Region')['count'].sum()[details_df.groupby('Region')['count'].sum() >= 2].index)]
        if len(excluded_regions) > 0:
            f.write(f"\nExcluded regions (<2 cells): {len(excluded_regions['Region'].unique())}\n")
            for region in excluded_regions['Region'].unique():
                cell_count = excluded_regions[excluded_regions['Region'] == region]['count'].sum()
                f.write(f"  • {region}: {cell_count} cells\n")
    
    print(f"Allen Brain Atlas mapping report saved: {report_path}")

def main():
    """Main analysis function"""
    print("Loading data...")
    details_df, summary_df = load_data()
    
    print("Filtering excluded regions (basic cell groups, claustrum, cerebral cortex)...")
    details_df = filter_excluded_regions(details_df)
    
    print("Filtering data for minimum cell threshold (2 cells)...")
    details_df = filter_minimum_cells(details_df)
    
    # Print detailed Allen Brain Atlas mapping information
    print_allen_mapping_info(details_df)
    
    # Print custom cortical grouping information
    print_custom_cortical_mapping_info(details_df)
    
    print("Creating enhanced visualizations...")
    
    # Apply custom cortical and subcortical groupings
    details_df = create_custom_cortical_groups(details_df)
    details_df = create_custom_subcortical_groups(details_df)
    
    # Create detailed mapping report file (after custom groupings are applied)
    create_allen_mapping_report(details_df)
    
    # Get cortical and subcortical data
    cortical_mask = details_df['is_cortical'] == True
    subcortical_mask = details_df['is_subcortical'] == True
    
    # Include ALL amygdalar regions in subcortical data and exclude from cortical
    amygdalar_mask = details_df['Region'].str.contains('amygdalar', case=False, na=False)
    # Also include specific amygdalar regions that might not be caught by the string search
    specific_amygdalar_regions = [
        'Cortical amygdalar area',
        'Piriform-amygdalar area'
    ]
    specific_amygdalar_mask = details_df['Region'].isin(specific_amygdalar_regions)
    
    # Combine both masks
    all_amygdalar_mask = amygdalar_mask | specific_amygdalar_mask
    
    subcortical_mask = subcortical_mask | all_amygdalar_mask
    cortical_mask = cortical_mask & ~all_amygdalar_mask  # Exclude amygdalar regions from cortical
    
    cortical_data = details_df[cortical_mask]
    subcortical_data = details_df[subcortical_mask]
    
    # 1. Subcortical custom groups heatmap
    print("Creating Custom Subcortical Groups Heatmap...")
    subcortical_heatmap_data = create_parent_group_heatmap(
        subcortical_data,
        "Custom Subcortical Groups per Brain (Normalized to 100% per Brain)",
        FIGURES_DIR / "2_custom_subcortical_groups_heatmap",
        'custom_subcortical_group',
        figsize=(10, 8)
    )
    
    # 2. Cortical custom groups heatmap
    print("Creating Custom Cortical Groups Heatmap...")
    cortical_heatmap_data = create_parent_group_heatmap(
        cortical_data,
        "Custom Cortical Groups per Brain (Normalized to 100% per Brain)",
        FIGURES_DIR / "3_custom_cortical_groups_heatmap",
        'custom_cortical_group',
        figsize=(12, 8)
    )
    
    # 3. Enhanced subcortical bar plot with individual values
    print("Creating Enhanced Custom Subcortical Bar Plot...")
    subcortical_stats, subcortical_individual = calculate_group_stats_with_individuals(
        subcortical_data, 'custom_subcortical_group'
    )
    create_enhanced_bar_plot(
        subcortical_stats,
        "Custom Subcortical Groups (Normalized to 100% within Subcortex) with Individual Brain Values",
        FIGURES_DIR / "2_custom_subcortical_regions_enhanced",
        subcortical_individual,
        figsize=(14, 10)
    )
    
    # 4. Enhanced cortical bar plot with individual values
    print("Creating Enhanced Custom Cortical Bar Plot...")
    cortical_stats, cortical_individual = calculate_group_stats_with_individuals(
        cortical_data, 'custom_cortical_group'
    )
    create_enhanced_bar_plot(
        cortical_stats,
        "Custom Cortical Groups (Normalized to 100% within Cortex) with Individual Brain Values",
        FIGURES_DIR / "3_custom_cortical_regions_enhanced",
        cortical_individual,
        figsize=(14, 10)
    )
    
    # 5. Create comprehensive Excel file with all graph data
    print("Creating comprehensive Excel file with all graph data...")
    create_comprehensive_excel(
        details_df, 
        summary_df,
        subcortical_heatmap_data=subcortical_heatmap_data,
        cortical_heatmap_data=cortical_heatmap_data,
        subcortical_stats=subcortical_stats,
        cortical_stats=cortical_stats,
        subcortical_individual=subcortical_individual,
        cortical_individual=cortical_individual
    )
    
    print(f"\nEnhanced analysis complete! Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Excel files saved to: {EXCEL_DIR}")
    print("\nAll figures saved as PNG, SVG, and PDF for Illustrator editing!")

if __name__ == "__main__":
    main()
