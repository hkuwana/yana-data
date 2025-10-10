# Brain Analysis Suite

This repository contains comprehensive analysis of brain region data with enhanced visualizations, individual brain value tracking, and complete data reproducibility. This project is open source and freely available under the MIT License.

## Overview

The analysis processes brain region data from multiple sources and creates detailed visualizations showing:
- Cortical vs subcortical distribution comparisons
- Thalamic subdivisions (polymodal vs sensory modal)
- Individual brain values for each region
- Normalized percentages within categories
- Comprehensive heatmaps and bar charts
- **Complete data tables for graph reproducibility**

## Files Structure

```
yana-data/
├── results/                   # Main analysis results
│   ├── figures/               # All visualizations (11 figures)
│   │   ├── 1_cortical_vs_subcortical_comparison.*
│   │   ├── 4_thalamus_polymodal_vs_sensory.*
│   │   ├── 5_polymodal_thalamus_subnuclei.*
│   │   ├── 6_sensory_thalamus_subnuclei.*
│   │   ├── 7_interlaminar_nuclei_polymodal.*
│   │   ├── 8_custom_subcortical_groups_heatmap.*
│   │   ├── 9_custom_cortical_groups_heatmap.*
│   │   ├── 10_custom_subcortical_regions_enhanced.*
│   │   └── 11_custom_cortical_regions_enhanced.*
│   ├── excel/                 # Comprehensive Excel files with all graph data
│   │   └── enhanced_brain_analysis.xlsx
│   ├── allen_brain_atlas_mapping_report.txt
│   ├── excluded_regions_filter.txt
│   └── excluded_regions_min_cells.txt
├── brain_analysis.py          # Main analysis script
├── organize_allen_data.py     # Data organization script
├── verify_heatmaps.py         # Utility script for verification
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## Generated Figures (All Normalized to 100%)

### 1. Cortical vs Subcortical Comparison
- **File**: `1_cortical_vs_subcortical_comparison.*`
- **Description**: Bar plot comparing cortical vs subcortical distribution
- **Normalization**: Normalized to 100% of total cells across all brains
- **Features**: Shows percentage breakdown with total cell counts

### 2. Thalamus: Polymodal vs Sensory Modal
- **File**: `4_thalamus_polymodal_vs_sensory.*`
- **Description**: Bar plot showing polymodal vs sensory modal thalamus
- **Normalization**: Normalized to 100% within thalamus
- **Features**: Includes individual brain values and mean±SEM

### 3. Polymodal Thalamus Subnuclei
- **File**: `5_polymodal_thalamus_subnuclei.*`
- **Description**: Bar plot showing subnuclei of polymodal thalamus
- **Normalization**: Normalized to 100% within polymodal thalamus
- **Subdivisions**: Anterior group, Epithalamus, Intralaminar nuclei, Lateral group, Medial group, Midline group, Reticular nucleus

### 4. Sensory Modal Thalamus Subnuclei
- **File**: `6_sensory_thalamus_subnuclei.*`
- **Description**: Bar plot showing subnuclei of sensory modal thalamus
- **Normalization**: Normalized to 100% within sensory modal thalamus
- **Subdivisions**: Geniculate group, Subparafascicular nucleus, Ventral group

### 5. Interlaminar Nuclei of Polymodal Thalamus
- **File**: `7_interlaminar_nuclei_polymodal.*`
- **Description**: Bar plot showing interlaminar nuclei of polymodal thalamus
- **Normalization**: Normalized to 100% within interlaminar nuclei
- **Features**: Individual nuclei breakdown with individual brain values

### 6. Custom Subcortical Groups Heatmap
- **File**: `8_custom_subcortical_groups_heatmap.*`
- **Description**: Heatmap showing distribution of custom subcortical groups across individual brains
- **Normalization**: Each brain normalized to 100% within subcortical regions
- **Custom Groups**: Striatum, Thalamus, Hypothalamus, Pallidum, Amygdala, Midbrain

### 7. Custom Cortical Groups Heatmap
- **File**: `9_custom_cortical_groups_heatmap.*`
- **Description**: Heatmap showing distribution of custom cortical functional groups across individual brains
- **Normalization**: Each brain normalized to 100% within cortical regions
- **Custom Groups**: See detailed breakdown below

### 8. Enhanced Subcortical Bar Plot
- **File**: `10_custom_subcortical_regions_enhanced.*`
- **Description**: Bar chart showing mean ± SEM with individual brain values displayed
- **Features**: Individual brain percentages shown as text annotations, error bars, color-coded bars

### 9. Enhanced Cortical Bar Plot
- **File**: `11_custom_cortical_regions_enhanced.*`
- **Description**: Bar chart showing mean ± SEM with individual brain values displayed
- **Features**: Individual brain percentages shown as text annotations, error bars, color-coded bars

## Key Features

### 1. Custom Cortical Functional Groupings

The script implements custom cortical groupings based on functional areas and Allen Brain Atlas classifications:

- **Association Areas**: Association regions (Posterior parietal association areas, Temporal association areas, Retrosplenial area)
- **Auditory Areas**: Auditory regions (Primary auditory area, Dorsal auditory area, Ventral auditory area)
- **Gustatory Area**: Gustatory areas as separate functional group
- **Hippocampal Formation**: Hippocampal regions (Dentate gyrus, Entorhinal area, Field CA1/CA2/CA3, Subiculum, etc.)
- **Insular Areas**: Insular regions (Agranular insular area)
- **Medial Prefrontal Cortex**: Specific MPFC regions (Anterior cingulate area, Infralimbic area, Prelimbic area, Orbital area)
- **Olfactory Areas**: Olfactory regions (Anterior olfactory nucleus, Main olfactory bulb, Piriform area, etc.)
- **Perirhinal Areas**: Perirhinal/Ectorhinal regions (Perirhinal area, Ectorhinal area layers)
- **Somatomotor Area**: Motor areas (Primary motor area, Secondary motor area)
- **Somatosensory Cortex**: Somatosensory regions from isocortex (Primary somatosensory area, Supplemental somatosensory area, Visceral area)
- **Visual Areas**: All visual areas separated from isocortex (Primary visual area, Anterolateral visual area, Anteromedial visual area)

**Note**: All amygdalar regions are now properly grouped under "Amygdala" in the subcortical analysis, not cortical.

### 2. Allen Brain Atlas Mapping Transparency

The script provides complete transparency about how each brain region is mapped to Allen Brain Atlas groups:

- **Console Output**: Detailed mapping information printed during script execution
- **Mapping Report**: Comprehensive text file (`allen_brain_atlas_mapping_report.txt`) with:
  - Allen Brain Atlas ID, acronym, and official name for each region
  - Parent group assignments (e.g., Hypothalamic medial zone, Isocortex)
  - Cell counts and brain presence statistics
  - Complete hierarchical organization including thalamic subdivisions

**Example Mapping Information:**
```
THALAMUS, POLYMODAL ASSOCIATION CORTEX RELATED:
  Anterior group of the dorsal thalamus:
    • Anterodorsal nucleus (ID: 64, Acronym: AD)
    • Anteromedial nucleus (ID: 127, Acronym: AM)
    • Anteroventral nucleus of thalamus (ID: 255, Acronym: AV)
```

### 3. Complete Data Reproducibility

#### Excel File: `enhanced_brain_analysis.xlsx` (21 Comprehensive Sheets)

**Core Data Sheets:**
- **Sheet 1: Raw_Data_Summary** - Summary statistics for all regions across all brains
- **Sheet 2: Complete_Raw_Data** - Complete processed data for full reproducibility

**New Figure Data Sheets:**
- **Sheet 12: Cortical_vs_Subcortical** - Data for cortical vs subcortical comparison
- **Sheet 13: Thalamus_Stats** - Summary statistics for thalamus polymodal vs sensory
- **Sheet 14: Thalamus_Individual_Brains** - Individual brain values for thalamus
- **Sheet 15: Polymodal_Thalamus_Stats** - Summary statistics for polymodal thalamus subnuclei
- **Sheet 16: Polymodal_Thalamus_Individual** - Individual brain values for polymodal thalamus
- **Sheet 17: Sensory_Thalamus_Stats** - Summary statistics for sensory thalamus subnuclei
- **Sheet 18: Sensory_Thalamus_Individual** - Individual brain values for sensory thalamus
- **Sheet 19: Interlaminar_Nuclei_Stats** - Summary statistics for interlaminar nuclei
- **Sheet 20: Interlaminar_Nuclei_Individual** - Individual brain values for interlaminar nuclei

**Original Figure Data Sheets:**
- **Sheet 3: Subcortical_Heatmap_Data** - Exact data used for subcortical groups heatmap
- **Sheet 4: Cortical_Heatmap_Data** - Exact data used for cortical groups heatmap
- **Sheet 5: Subcortical_BarPlot_Stats** - Summary statistics (mean±SEM) for subcortical bar plot
- **Sheet 6: Cortical_BarPlot_Stats** - Summary statistics (mean±SEM) for cortical bar plot
- **Sheet 7: Subcortical_Individual_Brains** - Individual brain values for subcortical bar plot
- **Sheet 8: Cortical_Individual_Brains** - Individual brain values for cortical bar plot

**Raw Data Sheets:**
- **Sheet 9: Subcortical_Raw_Data** - Complete raw data used for subcortical graphs
- **Sheet 10: Cortical_Raw_Data** - Complete raw data used for cortical graphs
- **Sheet 11: All_Regions_Individual** - Individual brain values for all regions (raw counts)

**Reference Sheet:**
- **Sheet 21: Graph_Data_Guide** - Instructions showing which sheets contain data for each graph

**Additional Files:**
- `allen_brain_atlas_mapping_report.txt`: Complete mapping of all regions to Allen Brain Atlas groups
- `excluded_regions_filter.txt`: List of regions excluded by user request
- `excluded_regions_min_cells.txt`: List of regions excluded due to low cell counts

## Data Processing

### Normalization Strategy

1. **Within-Brain Normalization**: Each brain's values are normalized to 100% within cortical or subcortical categories
2. **Within-Group Normalization**: Bar charts show percentages normalized within each parent group
3. **Minimum Cell Threshold**: Regions with <2 cells across all brains are excluded

### Quality Control

- **Minimum Brain Threshold**: Groups must be present in at least 3 brains to be included
- **Cell Count Filtering**: Regions with insufficient cell counts (<2 cells) are excluded and logged
- **Region Filtering**: Specific regions are excluded by user request:
  - Basic cell groups and regions
  - Claustrum
  - Cerebral cortex
  - Endopiriform nucleus
  - Non-amygdalar regions with "Other" allen_primary_group
- **Amygdala Preservation**: Amygdalar regions with "Other" allen_primary_group are kept and properly grouped under "Amygdala"
- **Data Validation**: All calculations include error checking and validation

## Usage

### For Complete Beginners (Step-by-Step Guide)

If you're new to GitHub and Python, follow these steps carefully:

#### Step 1: Download Python
1. Go to [python.org](https://www.python.org/downloads/)
2. Click "Download Python" (latest version)
3. Run the installer
4. **IMPORTANT**: Check the box "Add Python to PATH" during installation
5. Click "Install Now"

#### Step 2: Download This Project from GitHub
1. Go to this GitHub page
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to your Desktop (or any folder you prefer)
5. You should now have a folder called `yana-data-main` (or similar)

#### Step 3: Install Required Python Packages
1. Press `Windows + R` (or `Cmd + Space` on Mac)
2. Type `cmd` and press Enter (this opens Command Prompt)
3. Type these commands one by one, pressing Enter after each:
   ```
   pip install pandas
   pip install numpy
   pip install matplotlib
   pip install seaborn
   pip install openpyxl
   ```
   Wait for each to finish before typing the next one.

#### Step 4: Navigate to the Project Folder
1. In Command Prompt, type: `cd Desktop\yana-data-main` (Windows) or `cd Desktop/yana-data-main` (Mac)
2. Press Enter
3. If you put the folder somewhere else, adjust the path accordingly

#### Step 5: Run the Analysis
1. Type: `python brain_analysis.py`
2. Press Enter
3. Wait for the analysis to complete (it will show progress messages)

#### Step 6: Find Your Results
1. Look for a new folder called `results` in your project folder
2. Inside you'll find:
   - `figures/` folder with all 11 graphs (PNG, SVG, PDF formats)
   - `excel/` folder with the Excel file containing 21 comprehensive sheets
   - `allen_brain_atlas_mapping_report.txt` with detailed mapping information

#### Troubleshooting for Beginners

**If you get "python is not recognized":**
- Python wasn't added to PATH during installation
- Reinstall Python and make sure to check "Add Python to PATH"

**If you get "pip is not recognized":**
- Try: `python -m pip install pandas` instead of `pip install pandas`

**If you get "No such file or directory":**
- Make sure you're in the right folder
- Type `dir` (Windows) or `ls` (Mac) to see what files are there
- You should see `brain_analysis.py`

**If you get permission errors:**
- Try running Command Prompt as Administrator (right-click → "Run as administrator")

### For Users with Python Experience

#### Running the Main Analysis

```bash
python3 brain_analysis.py
```

#### Data Organization (if needed)

If you need to process raw data first:

```bash
python3 organize_allen_data.py
```

#### Verification

To verify heatmap regions:

```bash
python3 verify_heatmaps.py
```

#### Output Files

All visualizations are saved in three formats:
- **PNG**: High-resolution raster images (300 DPI)
- **SVG**: Vector graphics for editing
- **PDF**: Print-ready vector graphics

#### Data Requirements

The script will automatically look for data files in the following order:
1. `results/allen_region_details.csv`: Detailed region data with individual brain values
2. `results/allen_region_summary.csv`: Summary statistics
3. `results/excel/enhanced_brain_analysis.xlsx`: Existing Excel file (if CSV files don't exist)

If these files don't exist, you may need to run `organize_allen_data.py` first to generate them from your raw data.

## Key Improvements

### 1. Comprehensive Thalamic Analysis
- **Polymodal vs Sensory Modal**: Clear comparison of thalamic functional groups
- **Subnuclei Breakdown**: Detailed analysis of thalamic subdivisions
- **Interlaminar Nuclei**: Specific focus on polymodal thalamic interlaminar nuclei
- **Individual Brain Values**: Complete transparency of data across brains

### 2. Enhanced Visualizations
- **11 Total Figures**: Comprehensive coverage of brain regions
- **Multiple Formats**: PNG, SVG, and PDF for all figures
- **Normalized to 100%**: All figures properly normalized within categories
- **Individual Brain Tracking**: Each graph shows individual brain values

### 3. Complete Data Reproducibility
- **21 comprehensive Excel sheets** with all graph data
- **Exact data used for each graph** saved as separate sheets
- **Complete raw data** for full reproducibility
- **Graph Data Guide** showing which sheets contain data for each visualization
- **Individual brain values** preserved for all analyses

## Technical Details

### Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl (for Excel export)

### Data Columns Used
- `Region`: Specific brain region name
- `source_sheet`: Brain identifier (brain1, brain2, etc.)
- `count_pct_brain`: Percentage of cells in that brain
- `allen_primary_group`: Subcortical parent group
- `allen_cortical_class`: Cortical parent group
- `thalamic_group`: Thalamic functional group (polymodal vs sensory modal)
- `thalamic_subdivision`: Thalamic subdivision
- `is_cortical`/`is_subcortical`: Boolean flags for categorization

### Statistical Methods
- **Mean**: Average percentage across brains
- **SEM**: Standard Error of the Mean (std/sqrt(n))
- **Normalization**: Sum-to-100% within categories
- **Filtering**: Minimum thresholds for inclusion

## Results Interpretation

### Bar Charts
- **Bar height**: Mean percentage within group
- **Error bars**: Standard error of the mean
- **Individual values**: Shown as text annotations to the right
- **Color coding**: Different colors for different groups

### Heatmaps
- **Color intensity**: Higher percentages (darker colors)
- **Row order**: Sorted by mean percentage (highest to lowest)
- **Column order**: Individual brains (brain1, brain2, etc.)

### Excel Tables
- **Normalized sheets**: Values sum to 100% per brain
- **Raw sheets**: Original percentage values
- **Summary sheets**: Statistical measures across brains

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows you to:
- ✅ Use the software for any purpose
- ✅ Modify and distribute the software
- ✅ Use the software in commercial projects
- ✅ Distribute copies of the software
- ✅ Include the software in proprietary software

The only requirement is that you include the original copyright notice and license text in any copies or substantial portions of the software.

## Contact

For questions about this analysis or data processing methods, please refer to the code comments in the Python scripts or examine the Excel output files for detailed data breakdowns.