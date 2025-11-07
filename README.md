# NNI-pred: Neonicotinoid Insecticide Multimedia Transport Prediction

## Project Overview

This project explores machine learning models to predict neonicotinoid insecticide multimedia transport in basins. The prediction is based on soil pollutant concentrations and related environmental variables from both soil and water systems.

## Current Progress

✅ **Exploratory Data Analysis (EDA) Completed**
- Comprehensive statistical analysis of soil (186 samples, 51 features) and water (159 samples, 58 features) datasets
- Identified key correlations between agricultural activities and pollutant concentrations
- Analyzed seasonal variations, land use effects, and economic factors
- Data quality assessment confirmed excellent data integrity (92-94% quality score)

## Key Findings from EDA

- **Pollutant Distribution**: DIN dominates water pollutants (67%), while CLO leads in soil (37%)
- **Agricultural Impact**: Fertilizer/pesticide usage shows strong positive correlation with pollutant concentrations
- **Environmental Factors**: pH, dissolved oxygen, and water temperature significantly influence pollutant levels
- **Spatial Patterns**: Farmland areas show significantly higher pollutant concentrations than mountain/urban areas

## Project Structure

```
NNI-pred/
├── src/nni_pred/          # ML models and data processing code
├── scripts/               # Analysis and training scripts
├── docs/                  # Documentation and reports
├── tests/                 # Test files
└── outputs/               # Generated analysis outputs
```

## Dataset

- **Target**: Water body neonicotinoid concentrations (THIA, IMI, CLO, ACE, DIN)
- **Features**: Soil pollutant levels, environmental variables, agricultural factors, economic indicators
- **Geographic Coverage**: Longitude 112°-116°, Latitude 32.5°-34.8°

## Next Steps

- 🔄 Data preprocessing and feature engineering
- ⏳ Machine learning model development
- ⏳ Model validation and performance evaluation

## Documentation

- [Data Exploration Report](docs/data_exploration_report.md) - Detailed statistical analysis and insights

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
