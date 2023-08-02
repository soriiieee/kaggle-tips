
from scipy import stats
from scipy.stats import norm, skew, kurtosis

from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import catboost as cb
import xgboost as xgb
import inspect
from collections import defaultdict
from tabpfn import TabPFNClassifier

import warnings
warnings.filterwarnings('ignore')



from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings('ignore')

import numpy as np



# Create a LabelEncoder object.

class PreProcess:
    def __init__():
        self.encoder = LabelEncoder()
        encoder = LabelEncoder()
        # Transform the data.
    
    def label_encode(self,df,cat_cols):
        
        train[cat_cols] = self.encoder.fit_transform(df[cat_cols])
        test[cat_cols] = self.encoder.transform(test[cat_cols])
        return ain[cat_cols]


class CrossValidate:
    """
    参考：
    https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = svm.SVC(kernel='linear', C=1, random_state=42)
    >>> scores = cross_val_score(model, X, y, cv=5)
    >>> scores
    
    """
class CrossValidate:
    def __init__(self):
        print("start")
        
    def n_split(self):
        
        oof = np.zeros((len(train), 2))

        skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        final_preds = []

        params={
            'iterations':10000,
            'learning_rate':0.005,
            'early_stopping_rounds':1000,
            'auto_class_weights':'Balanced',
            'loss_function':'MultiClass',
            'eval_metric':'MultiClass:use_weights=True',
            'random_seed':42,
            'use_best_model':True,
            'l2_leaf_reg':1,
            'max_ctr_complexity':15,
            'max_depth':10,
            "grow_policy":'Lossguide',
            'max_leaves':64,
            "min_data_in_leaf":40,

            }

        for train_index,val_index in skf.split(train, greeks.iloc[:,1:-1]):

            X_train, X_val = train.loc[train_index, num_cols + [cat_cols]], train.loc[val_index, num_cols + [cat_cols]]
            y_train, y_val = train.loc[train_index, 'Class'], train.loc[val_index, 'Class']
            
            
            model = cb.CatBoostClassifier(**params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)], verbose=1000)
            preds = model.predict_proba(X_val)
            oof[val_index, :] = preds
            final_preds.append(model.predict_proba(test.iloc[:,1:]))

     
class Ensemble():
    def __init__(self):
        self.imputer = SimpleImputer(missing_values= np.nan, strategy='median')

        self.classifiers = [
            xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.2,
                                  subsample=0.9,colsample_bytree=0.85),
            xgb.XGBClassifier(),
            TabPFNClassifier(N_ensemble_configurations=24),
            TabPFNClassifier(N_ensemble_configurations=64)
            ]
    
    def fit(self,X,y):
        y = y.values
        unique_classes, y = np.unique(y, return_inverse=True)
        self.classes_ = unique_classes
        
        # first_category = X.EJ.unique()[0]
        # X.EJ = X.EJ.eq(first_category).astype('int')
        X = self.imputer.fit_transform(X)
#         X = normalize(X,axis=0)
        for classifier in self.classifiers:
            if classifier==self.classifiers[2] or classifier==self.classifiers[3]:
                classifier.fit(X,y,overwrite_warning =True)
            else :
                classifier.fit(X, y)
     
    def predict_proba(self, x):
        x = self.imputer.transform(x)
#         x = normalize(x,axis=0)
        probabilities = np.stack([classifier.predict_proba(x) for classifier in self.classifiers])
        averaged_probabilities = np.mean(probabilities, axis=0)
        class_0_est_instances = averaged_probabilities[:, 0].sum()
        others_est_instances = averaged_probabilities[:, 1:].sum()
        # Weighted probabilities based on class imbalance
        new_probabilities = averaged_probabilities * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(averaged_probabilities.shape[1])]])
        return new_probabilities / np.sum(new_probabilities, axis=1, keepdims=1) 