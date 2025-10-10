# Enhanced Brain Analysis

This repository contains comprehensive analysis of brain region data with enhanced visualizations and individual brain value tracking. This project is open source and freely available under the MIT License.

## Overview

The analysis processes brain region data from multiple sources and creates detailed visualizations showing:
- Parent group distributions for cortical and subcortical regions
- Individual brain values for each region
- Normalized percentages within categories
- Comprehensive heatmaps and bar charts

## Files Structure

```
yana-data/
├── enhanced_results/           # Enhanced analysis results
│   ├── figures/               # Enhanced visualizations
│   │   ├── 2_custom_subcortical_groups_heatmap.*
│   │   ├── 3_custom_cortical_groups_heatmap.*
│   │   ├── 2_custom_subcortical_regions_enhanced.*
│   │   └── 3_custom_cortical_regions_enhanced.*
│   ├── excel/                 # Comprehensive Excel files
│   │   └── enhanced_brain_analysis.xlsx
│   ├── allen_brain_atlas_mapping_report.txt
│   ├── excluded_regions_filter.txt
│   └── excluded_regions_min_cells.txt
├── comprehensive_results/      # Original comprehensive analysis
├── results/                   # Basic analysis results
├── enhanced_brain_analysis.py # Enhanced analysis script
├── comprehensive_brain_analysis.py # Original analysis script
├── README.md                  # This file
└── LICENSE                    # MIT License
```

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

The script also provides complete transparency about how each brain region is mapped to Allen Brain Atlas groups:

- **Console Output**: Detailed mapping information printed during script execution
- **Mapping Report**: Comprehensive text file (`allen_brain_atlas_mapping_report.txt`) with:
  - Allen Brain Atlas ID, acronym, and official name for each region
  - Parent group assignments (e.g., Hypothalamic medial zone, Isocortex)
  - Cell counts and brain presence statistics
  - Complete hierarchical organization

**Example Mapping Information:**
```
HYPOTHALAMUS:
  • Anterior hypothalamic nucleus (ID: 88, Acronym: AHN)
    Allen Name: Anterior hypothalamic nucleus
    Allen Parent: Hypothalamic medial zone
    Total Cells: 35
    Present in Brains: 4
```

### 3. Enhanced Visualizations

#### Custom Subcortical Groups Heatmap
- **File**: `2_custom_subcortical_groups_heatmap.*`
- **Description**: Heatmap showing distribution of custom subcortical groups across individual brains
- **Normalization**: Each brain normalized to 100% within subcortical regions
- **Custom Groups**: Striatum, Thalamus, Hypothalamus, Pallidum, Amygdala, Midbrain

#### Custom Cortical Groups Heatmap
- **File**: `3_custom_cortical_groups_heatmap.*`
- **Description**: Heatmap showing distribution of custom cortical functional groups across individual brains
- **Normalization**: Each brain normalized to 100% within cortical regions
- **Custom Groups**: 
  - **Association Areas**: Posterior parietal association areas, Temporal association areas, Retrosplenial area
  - **Auditory Areas**: Primary auditory area, Dorsal auditory area, Ventral auditory area
  - **Gustatory Area**: Gustatory areas
  - **Hippocampal Formation**: Dentate gyrus, Entorhinal area, Field CA1/CA2/CA3, Subiculum, etc.
  - **Insular Areas**: Agranular insular area
  - **Medial Prefrontal Cortex**: Anterior cingulate area, Infralimbic area, Prelimbic area, Orbital area
  - **Olfactory Areas**: Anterior olfactory nucleus, Main olfactory bulb, Piriform area, etc.
  - **Perirhinal Areas**: Perirhinal area, Ectorhinal area layers
  - **Somatomotor Area**: Primary motor area, Secondary motor area
  - **Somatosensory Cortex**: Primary somatosensory area, Supplemental somatosensory area, Visceral area
  - **Visual Areas**: Primary visual area, Anterolateral visual area, Anteromedial visual area

#### Enhanced Bar Charts with Individual Values
- **Files**: `2_subcortical_regions_enhanced.*` and `3_custom_cortical_regions_enhanced.*`
- **Description**: Bar charts showing mean ± SEM with individual brain values displayed
- **Features**: 
  - Individual brain percentages shown as text annotations
  - Error bars representing standard error of the mean
  - Color-coded bars for different groups

### 4. Comprehensive Data Tables

#### Excel File: `enhanced_brain_analysis.xlsx`

**Sheet 1: Raw_Data_Summary**
- Summary statistics for all regions across all brains

**Sheet 2: Custom_Subcortical_Groups_Individual**
- Individual brain values for custom subcortical groups (normalized to 100% per brain)

**Sheet 3: Custom_Cortical_Groups_Individual**
- Individual brain values for custom cortical functional groups (normalized to 100% per brain)

**Sheet 4: Custom_Subcortical_Raw_Individual**
- Raw individual brain values for custom subcortical groups (not normalized)

**Sheet 5: Custom_Cortical_Raw_Individual**
- Raw individual brain values for custom cortical groups (not normalized)

**Sheet 6: Custom_Subcortical_Summary_Stats**
- Summary statistics (mean, SEM, brain count, total cells) for custom subcortical groups

**Sheet 7: Custom_Cortical_Summary_Stats**
- Summary statistics (mean, SEM, brain count, total cells) for custom cortical groups

**Sheet 8: All_Regions_Individual**
- Individual brain values for all regions (raw counts)

**Additional Files:**
- `allen_brain_atlas_mapping_report.txt`: Complete mapping of all regions to Allen Brain Atlas groups
- `excluded_regions_filter.txt`: List of regions excluded by user request (basic cell groups, claustrum, cerebral cortex, endopiriform nucleus, and non-amygdalar "Other" regions)
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
1. Type: `python enhanced_brain_analysis.py`
2. Press Enter
3. Wait for the analysis to complete (it will show progress messages)

#### Step 6: Find Your Results
1. Look for a new folder called `enhanced_results` in your project folder
2. Inside you'll find:
   - `figures/` folder with all the graphs (PNG, SVG, PDF formats)
   - `excel/` folder with the Excel file
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
- You should see `enhanced_brain_analysis.py`

**If you get permission errors:**
- Try running Command Prompt as Administrator (right-click → "Run as administrator")

### For Users with Python Experience

#### Running the Enhanced Analysis

```bash
python3 enhanced_brain_analysis.py
```

#### Output Files

All visualizations are saved in three formats:
- **PNG**: High-resolution raster images (300 DPI)
- **SVG**: Vector graphics for editing
- **PDF**: Print-ready vector graphics

#### Data Requirements

- `results/allen_region_details.csv`: Detailed region data with individual brain values
- `results/allen_region_summary.csv`: Summary statistics

## Key Improvements

### 1. Individual Brain Tracking
- Each graph now shows individual brain values as text annotations
- Raw and normalized values available in Excel files
- Complete transparency of data processing

### 2. Enhanced Heatmaps
- Parent group level heatmaps for better overview
- Clear visualization of brain-to-brain variability
- Proper normalization within categories

### 3. Comprehensive Documentation
- Detailed README with file descriptions
- Clear explanation of normalization strategies
- Usage instructions and data requirements

### 4. Data Export
- Multiple Excel sheets with different views of the data
- Both raw and normalized values available
- Summary statistics for all groups

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
- `is_cortical`/`is_subcortical`: Boolean flags for categorization

### Statistical Methods
- **Mean**: Average percentage across brains
- **SEM**: Standard Error of the Mean (std/sqrt(n))
- **Normalization**: Sum-to-100% within categories
- **Filtering**: Minimum thresholds for inclusion

## Results Interpretation

### Heatmaps
- **Color intensity**: Higher percentages (darker colors)
- **Row order**: Sorted by mean percentage (highest to lowest)
- **Column order**: Individual brains (brain1, brain2, etc.)

### Bar Charts
- **Bar height**: Mean percentage within group
- **Error bars**: Standard error of the mean
- **Individual values**: Shown as text annotations to the right
- **Color coding**: Different colors for different groups

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
