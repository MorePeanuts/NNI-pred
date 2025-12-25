"""
Perform a simplest nested spatial cross-validation using the random forest model.
"""

import numpy as np
from nni_pred.models import RandomForestBuilder
from nni_pred.data import MergedTabularDataset
from nni_pred.transformers import GroupedPCA, TargetTransformer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer
from loguru import logger


dataset = MergedTabularDataset()
builder = RandomForestBuilder()
model = TransformedTargetRegressor(
    regressor=builder.get_regressor(), transformer=TargetTransformer()
)
X, y_dict, groups = dataset.prepare_data()
param_grid = builder.get_default_param_grid('small')

for _, (target, y) in enumerate(y_dict.items()):
    logger.info(f'Training Random Forest Predictor for {target}...')
    outer_cv = GroupKFold(5, shuffle=True, random_state=42)

    for train_val_idx, test_idx in outer_cv.split(X, y, groups):
        train_val_X = X.iloc[train_val_idx]
        train_val_y = y.iloc[train_val_idx]
        test_X = X.iloc[test_idx]
        test_y = y.iloc[test_idx]
        train_groups = groups[train_val_idx]

        pca = GroupedPCA(random_state=42)
        feature_engineering = ColumnTransformer(
            transformers=[('pca', pca, pca.get_feature_cols())], remainder='passthrough'
        )
        pipeline = Pipeline(
            [
                ('prep', feature_engineering),
                ('model', model),
            ]
        )

        param_grid_pipeline = {f'model__regressor__{k}': v for k, v in param_grid.items()}
        inner_cv = GroupKFold(4, shuffle=True, random_state=42)
        grid_search_cv = GridSearchCV(pipeline, param_grid_pipeline, scoring='r2', cv=inner_cv)

        grid_search_cv.fit(train_val_X, train_val_y, groups=train_groups)
        test_y_pred = grid_search_cv.predict(test_X)
        offset = grid_search_cv.best_estimator_.named_steps['model'].transformer_.offset_

        # Calculate the test set R2 on a log scale.
        r2 = r2_score(np.log(test_y.values + offset), np.log(test_y_pred + offset))
        logger.info(f'  - Test R2 for {target}: {r2}')
