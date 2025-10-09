#!/usr/bin/env python3
"""Quick verification script to show what regions are in each heatmap"""

import pandas as pd
from pathlib import Path

# Load data
DATA_DIR = Path("/Users/hiro/Documents/yana-data")
details_path = DATA_DIR / "results" / "allen_region_details.csv"
details_df = pd.read_csv(details_path)

# Filter for minimum 2 cells
region_counts = details_df.groupby('Region')['count'].sum()
valid_regions = region_counts[region_counts >= 2].index
details_df = details_df[details_df['Region'].isin(valid_regions)]

# Get cortical and subcortical masks
cortical_mask = details_df['is_cortical'] == True
subcortical_mask = details_df['is_subcortical'] == True

print("=== CORTEX HEATMAP REGIONS ===")
cortex_regions = details_df[cortical_mask]['Region'].unique()
print(f"Total cortical regions: {len(cortex_regions)}")
for region in sorted(cortex_regions):
    print(f"  - {region}")

print("\n=== SUBCORTEX HEATMAP REGIONS ===")
subcortex_regions = details_df[subcortical_mask]['Region'].unique()
print(f"Total subcortical regions: {len(subcortex_regions)}")
for region in sorted(subcortex_regions):
    print(f"  - {region}")

print("\n=== SUBCORTEX REGIONS BY PRIMARY GROUP ===")
subcortex_groups = details_df[subcortical_mask].groupby('allen_primary_group')['Region'].unique()
for group, regions in subcortex_groups.items():
    print(f"\n{group} ({len(regions)} regions):")
    for region in sorted(regions):
        print(f"  - {region}")
