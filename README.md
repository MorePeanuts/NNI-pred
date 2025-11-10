# NNI-pred: Neonicotinoid Insecticide Multimedia Transport Prediction

## Project Overview

This project develops machine learning models to predict neonicotinoid insecticide concentrations in water bodies based on soil pollutant concentrations and environmental variables. The research focuses on understanding and predicting the multimedia transport of neonicotinoid insecticides from soil to water systems.

## Research Goal

**Primary Objective**: Build accurate predictive models for water body pollutant concentrations based on soil monitoring data and environmental covariates.

## Current Stage: Stage 1 Completed ✅

### ✅ Core Achievements

**1. Comprehensive Data Analysis**
- Completed exploratory data analysis for both soil (186 samples) and water (159 samples) datasets
- Generated detailed statistical reports and visualizations
- Confirmed data quality and completeness

**2. Pollutant Correlation Analysis**
- Discovered critical insight: 85.4% of pollutant pairs show weak correlation (|r| ≤ 0.5)
- Only 3 strong correlation pairs identified: THIA-IMI, DIN-parentNNIs, DN-IMI-mNNIs
- **Strategic Pivot**: Shifted from mixed-model approach to 11 individual single-output models (primary) + 8 hybrid models (validation)

**3. Spatiotemporal Distribution Analysis**
- Analyzed geographic overlap and sample density patterns
- Generated comprehensive distance statistics and density maps
- Provided solid foundation for Stage 2 grid pairing algorithm development

### 📊 Key Reports Generated

- **[Stage 1 Correlation Analysis Report](docs/stage1_correlation_analysis.md)**
- **[Stage 1 Spatiotemporal Analysis Report](docs/stage1_spatiotemporal_analysis.md)**
- **[Stage 1 Summary Report](docs/stage1_summary_report.md)**

## Data Overview

- **Soil Dataset**: 186 samples with 51 features including pollutant concentrations, environmental factors, and agricultural variables
- **Water Dataset**: 159 samples with 58 features covering neonicotinoid concentrations and water chemistry parameters
- **Geographic Coverage**: Longitude 112°-116°, Latitude 32.5°-34.8°
- **Seasonal Balance**: Perfectly balanced sampling across Dry, Normal, Rainy seasons

## Key Scientific Insights

### 🎯 Critical Finding: Weak Correlation Dominance
- **85.4%** of pollutant pairs show weak correlation, challenging the effectiveness of multi-output models
- **Implication**: Individual pollutant modeling may be more appropriate than grouped approaches

### 📊 Strong Correlation Pairs Identified
- **THIA ↔ IMI**: r = 0.94 (potential common usage sources)
- **DIN ↔ parentNNIs**: r = 0.94 (composition relationship)
- **DN-IMI ↔ mNNIs**: r = 0.94 (metabolite formation pattern)

## Next Stage: Stage 2 - In Progress 🔄

### 🎯 Primary Focus: Grid Pairing Algorithm
- **Objective**: Design adaptive grid-based pairing algorithm for soil-water sample association
- **Strategy**: Grid division centered on soil sample points to ensure clear sample attribution
- **Reference Parameters**: 0.2° (22.2 km) grid size, ~264 soil grid cells, ~240 water grid cells

### 📋 Key Technical Challenges
1. **Grid Definition**: Optimize grid size based on sample density and spatial distribution
2. **Pairing Constraints**: Ensure maximum one water sample per grid, minimum one soil sample when water present
3. **Seasonal Consistency**: Strict seasonal separation for pairing operations
4. **Distance Weighting**: Implement inverse distance weighting for multiple soil samples per grid

## Project Structure

```
NNI-pred/
├── src/nni_pred/          # Core ML models and data processing modules
├── scripts/               # Analysis and training scripts
│   ├── correlation_analysis.py     # Pollutant correlation analysis
│   ├── spatiotemporal_analysis.py # Spatial distribution analysis
│   └── explore_data.py              # Exploratory data analysis
├── docs/                  # Documentation and reports
│   ├── data_exploration_report.md   # Initial EDA report
│   ├── research_plan.md             # Research plan (this file)
│   ├── stage1_correlation_analysis_report.md
│   ├── stage1_spatiotemporal_analysis_report.md
│   └── stage1_summary_report.md
├── outputs/               # Generated analysis outputs
│   ├── stage1_correlation_analysis/  # Correlation analysis results
│   │   ├── correlation_matrices/
│   │   ├── visualizations/
│   │   └── pollutant_grouping.json
│   └── stage1_spatiotemporal_analysis/  # Spatial analysis results
│       ├── spatial_statistics.csv
│       ├── grid_parameters.json
│       └── visualizations/
└── datasets/              # Data files
    ├── soil_data.csv           # Soil monitoring data
    ├── water_data.csv          # Water monitoring data
    └── raw_data.xlsx         # Original raw data
├── tests/                 # Test files (currently empty)
└── temp-py/              # Temporary scripts
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run Stage 1 correlation analysis
uv run scripts/correlation_analysis.py

# Run Stage 1 spatial analysis
uv run scripts/spatiotemporal_analysis.py
```

## Project Status

- **Stage 1**: ✅ Completed (Correlation & Spatial Analysis)
- **Stage 2**: 🔄 In Progress (Grid Pairing Algorithm)
- **Stage 3**: ⏳ Planned (Feature Engineering & Model Implementation)
- **Stage 4**: ⏳ Planned (Model Validation & Optimization)
- **Stage 5**: ⏳ Planned (Deployment & Application)

## Research Innovation

**Data-Driven Strategy Adjustment**:
- Initial assumption: Multi-output models leveraging pollutant correlations
- Data reality: Weak correlation dominates (85.4% of pairs)
- Revised approach: Individual single-output models with validation comparison

**Spatial Analysis Foundation**:
- Comprehensive sample distribution analysis provides solid foundation for grid design
- Seasonal and geographic consistency verified through statistical analysis
- Reference parameters established for adaptive grid algorithm development

---

**Project Status**: Ready for Stage 2 implementation with solid analytical foundation.

## Requirements

- Python 3.12+
- `uv` package manager

## Quick Start

```bash
# Install dependencies
uv sync

# Run exploratory analysis
uv run scripts/explore_data.py
```
