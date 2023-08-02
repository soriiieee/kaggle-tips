# laggle set
import os
import json
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import optuna.integration.lightgbm as lgb


@dataclass(frozen=True)
class Config:
    n_folds: int = 8
    random_seed: int = 42 

@dataclass
class ParamsLGBM:
    objective: str = "binary"
    metric: str = "binary_logloss"
    verbosity: int = -1
    is_unbalance: bool = True
    boosting_type: str = "gbdt"
    learning_rate: float = 0.01
    max_depth: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.63
    lambda_l2: float = 6.6
    
    def dict(self) -> dict:
        return asdict(self)
    

class TrainDataSet:
    def __init__(self, config, df: pd.DataFrame):
        from imblearn.over_sampling import RandomOverSampler
        
        self.non_feat_cols = [
            'Id', # ID
            'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', # Greeks
            'fold', 'oof', # Artifacts
            'DV', 'BZ', 'CL', 'CH', 'EL', 'EP', 'EG' # Unimportant
        ]
        self.target_col = ["Class"]
        self.df = df
        self.df.columns = self.df.columns.str.strip()
        self.ros = RandomOverSampler(random_state= Config.random_seed)
        
        self._add_fold_col()
        self.df["EJ"] = self.df.EJ.map({"A":0, "B":1}).astype('int')
        
    def get_feature(self, fold:int, feature_name: str, phase: str='train') -> pd.Series:
        if phase == 'train':
            return self.df.loc[~self._is_val_target(fold), feature_name]
        else:
            return self.df.loc[self._is_val_target(fold), feature_name] 
    
    def _add_fold_col(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=Config.n_folds, random_state=Config.random_seed, shuffle=True)

        self.df["fold"] = np.nan
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df["Gamma"])):
            self.df.loc[val_idx, "fold"] = fold
            
    def _is_val_target(self, fold: int) -> pd.Series:
        return self.df.fold == fold
    
    def add_oof_preds(self, fold: int, oof_preds: np.array) -> None:
        self.df.loc[self._is_val_target(fold), "oof"] = oof_preds
    
    def targets(self, fold: int) -> (pd.DataFrame, pd.DataFrame):
        return self.df.loc[~self._is_val_target(fold), self.target_col], self.df.loc[self._is_val_target(fold), self.target_col]
    
    def inputs(self, fold: int) -> (pd.DataFrame, pd.DataFrame):
        input_cols = [f for f in self.df.columns if not f in self.target_col + self.non_feat_cols]
        return self.df.loc[~self._is_val_target(fold), input_cols], self.df.loc[self._is_val_target(fold), input_cols]

    def train_dataset_oversampled(self, fold: int, feature_name: str='Class') -> (pd.DataFrame, pd.DataFrame):
        oversampled_data, _ = self.ros.fit_resample(
            self.df.loc[~self._is_val_target(fold)], 
            self.df.loc[~self._is_val_target(fold),feature_name]
            )
        oversampled_df = pd.DataFrame(oversampled_data)
        input_cols = [f for f in oversampled_df.columns if not f in self.target_col + self.non_feat_cols]
        return oversampled_df.loc[:, input_cols], oversampled_df.loc[:, self.target_col]


DEBUG=True
class Trainer:
    
    def __init__(self,config,train_ds):
        
        self.config = config
        self.train_ds = train_ds
        self.debug = False
    
    def train(self,train_df,train_target,valid_df,valid_target):
        
        for fold in range(self.config.n_folds):
            train_input, val_input = self.train_ds.inputs(fold)
            train_target, val_target = self.train_ds.targets(fold)
            
        #     train_input, train_target = train_ds.train_dataset_oversampled(fold, 'Class')
            
            N0, N1 = np.bincount(train_df.Class) #509 108
            train_weight = train_target['Class'].map({0: 1/N0, 1: 1/N1})
            val_weight = val_target['Class'].map({0: 1/N0, 1: 1/N1})
            
            train_ds_lgb = lgb.Dataset(train_input, train_target, weight=train_weight)
            val_ds_lgb = lgb.Dataset(val_input, val_target, weight=val_weight, reference=train_ds_lgb)

            model = lgb.train(
                ParamsLGBM().dict(),
                train_ds_lgb, 
                num_boost_round=100000 if not self.debug else 1,
                valid_sets=[train_ds_lgb, val_ds_lgb], 
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(400),
                ]
            )
            model.save_model(f'lgbm_fold_{fold}.txt', num_iteration = model.best_iteration)
            preds = model.predict(val_input)
            
            self.train_ds.add_oof_preds(fold, preds)
            print(f'Best Params in fold {fold}:')
            pprint(model.params)
            with open(f'params_lgbm_fold_{fold}.json', 'w') as f:
                json.dump(model.params, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    print(ParamsLGBM().dict())