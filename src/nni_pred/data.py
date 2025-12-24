import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path


SOIL_POLLUTANTS = [
    'THIA',
    'IMI',
    'CLO',
    'parentNNIs',
    'IMI-UREA',
    'DN-IMI',
    'CLO-UREA',
    'mNNIs',
]
SOIL_SEASONAL_VARS = ['pH', 'TN', 'TOC', 'Temp', 'Rain', 'EVI', 'FCOVER', 'LAI', 'LST', 'NDVI']
SOIL_ANNUAL_VARS = [
    'Alt',
    'CC',
    'BD',
    'GPAM',
    'WS',
    'FER',
    'PES',
    'MC',
    'MCP',
    'VE',
    'VEP',
    'FR',
    'FRP',
    'FERA',
    'PESA',
    'UR',
    'GAO',
    'PR',
    'SR',
    'TR',
    'GDP per capita',
    'UI',
    'RI',
    'UR_W',
    'RU_W',
    'IRR_W',
    'AGR_W',
    'IND_W',
    'LIF_W',
]
SOIL_CATEGORICAL_VARS = ['landuse']


class MergedTabularDataset:
    pass
