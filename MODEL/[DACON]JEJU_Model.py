from flaml import AutoML
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from os.path import join
import os
import warnings
warnings.filterwarnings("ignore")


# we_train, we_test 날씨가 있는 데이터 사용
# 요일, 시작지점과 도착지점의 회전제한 유무, 도로명, 날씨 지역을 label encoder사용하여 수치형으로 변환
str_col = ['day_of_week', 'start_turn_restricted', 'end_turn_restricted',
           'road_name', 'end_node_name', 'start_node_name', 'stnNm']
for i in str_col:
    le = LabelEncoder()
    le = le.fit(we_train[i])
    we_train[i] = le.transform(we_train[i])

    for label in np.unique(we_test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    we_test[i] = le.transform(we_test[i])

# 날짜를 변환해서 년원일분으로 나눠주는 함수


def split_date(df, col):
    date = pd.to_datetime(df[col])
    df[col] = date
    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["day"] = date.dt.day
    df['year_month'] = date.dt.strftime('%Y%m')

# 시간 범주화


def time_category(x):
    if (0 <= x <= 5):
        y = 3
    if (6 <= x <= 7) or (22 <= x <= 23):
        y = 2
    if (8 <= x <= 16) or (19 <= x <= 21):
        y = 1
    if (17 <= x <= 18):
        y = 0
    return y


# 문자열로 변환
we_train["base_date"] = we_train["base_date"].astype(str)
we_test["base_date"] = we_test["base_date"].astype(str)

# train, test 데이터 둘 다 변환
split_date(we_train, "base_date")
split_date(we_test, "base_date")

we_train["year_month"] = we_train["year_month"].astype(int)
we_test["year_month"] = we_test["year_month"].astype(int)

we_train["time_category"] = we_train["base_hour"].apply(time_category)
we_test["time_category"] = we_test["base_hour"].apply(time_category)

#  6, 7월 가중치 2회씩 줌
train = pd.concat([we_train, we_train[we_train["month"] == 6],
                  we_train[we_train["month"] == 7]], ignore_index=True)


y_train = train['target']
# 날씨에서 null의 개수가 많은 컬럼 'sumRn', 'maxInsWs','maxWs', 'avgTca','avgLmac','sumFogDur' 제거
X_train = train.drop(['id', 'base_date', 'month', 'year', 'day', 'multi_linked', 'connect_code', 'height_restricted',
                     'vehicle_restricted', 'road_type', 'target', 'sumRn', 'maxInsWs', 'maxWs', 'avgTca', 'avgLmac', 'sumFogDur'], axis=1)

test = we_test.drop(['id', 'base_date', 'month', 'year', 'day', 'multi_linked', 'connect_code', 'height_restricted',
                    'vehicle_restricted', 'road_type', 'sumRn', 'maxInsWs', 'maxWs', 'avgTca', 'avgLmac', 'sumFogDur'], axis=1)

carbon_monoxide_predictor = AutoML()

settings = {
    "metric": 'mae',
    "estimator_list": ['lgbm'],  # 'xgboost', 'catboost', 'extra_tree'
    "task": 'regression',
    "log_file_name": "california.log",
    "time_budget": 5000
}

carbon_monoxide_predictor.fit(X_train, y_train, **settings)


print('Best estimator:', carbon_monoxide_predictor.best_estimator)
print('Best hyperparmeter config:', carbon_monoxide_predictor.best_config)
print('Training duration of best run: {0:.4g} s'.format(
    carbon_monoxide_predictor.best_config_train_time))

model_lgb = lgb.LGBMRegressor(learning_rate=0.030121482550763388, max_bin=1023,
                              min_child_samples=9, n_estimators=9201, num_leaves=2269,
                              reg_alpha=0.00706357318094864, reg_lambda=0.004210296104410366,
                              verbose=-1)
model_lgb.fit(X_train, y_train)
lgb_pred = model_lgb.predict(test)

model_Extra = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25,
                                  min_samples_leaf=35, max_features=15)
model_Extra.fit(X_train, y_train)
Extra_pred = model_Extra.predict(test)

xgb_model = XGBRegressor(n_estimators=500, random_state=seed,
                         max_depth=6,  objective='reg:squarederror',
                         )
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(test)


sample_submission = pd.read_csv('../input/jejuload-set/sample_submission.csv')
sample_submission['target'] = 0.8*xgb_pred + 0.15*lgb_pred + 0.05*Extra_pred
sample_submission.to_csv("./submit.csv", index=False)
