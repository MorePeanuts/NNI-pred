"""
Perform a simplest nested spatial cross-validation using the random forest model.
"""

from nni_pred.models import NNIPredictorRF
from nni_pred.data import MergedTabularDataset
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score
from loguru import logger


dataset = MergedTabularDataset()
model = NNIPredictorRF().model
X, y_dict, groups = dataset.prepare_data()
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [8, 15],  # Tree depth
    'min_samples_split': [5, 10],  # Min samples to split
    'min_samples_leaf': [2, 4],  # Min samples per leaf
    'max_features': ['sqrt', 'log2'],  # Features per split
}

for idx, (target, y) in enumerate(y_dict.items()):
    if idx >= 3:
        break
    logger.info(f'Training Random Forest Predictor for {target}...')
    outer_cv = GroupKFold(5, shuffle=True, random_state=42)

    for train_val_idx, test_idx in outer_cv.split(X, y, groups):
        train_val_X = X.iloc[train_val_idx]
        train_val_y = y.iloc[train_val_idx]
        test_X = X.iloc[test_idx]
        test_y = y.iloc[test_idx]
        train_groups = groups[train_val_idx]

        inner_cv = GroupKFold(4, shuffle=True, random_state=42)
        grid_search_cv = GridSearchCV(model, param_grid, scoring='r2', cv=inner_cv)

        grid_search_cv.fit(train_val_X, train_val_y, groups=train_groups)
        test_y_pred = grid_search_cv.predict(test_X)

        r2 = r2_score(test_y.values, test_y_pred)
        logger.info(f'  - Test R2 for {target}: {r2}')
