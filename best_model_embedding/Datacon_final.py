# %%
import pandas as pd
import json 
import numpy as np

# %%
df = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) 


# %%
df = df.dropna(subset=['Sequence'])
df.shape

# %%
df

# %%
with open("E:\\My_projects\\poseidon_mistral.json", "r") as f:
    embeddings_data = json.load(f)

# Извлечение эмбеддингов
data = []
for item in embeddings_data:
    input_value = item['input']
    embedding_values = item['embedding']
    # Разбиваем embedding на отдельные значения
    embedding_list = []  # Создаем пустой список для значений эмбеддинга
    for value in embedding_values:  # Итерируем по элементам списка embedding_values
        embedding_list.append(float(value))  # Добавляем значения в список embedding_list

    # Создаем словарь для записи в DataFrame
    row_data = {'input': input_value}
    for i, value in enumerate(embedding_list):
        row_data[f'embedding_{i+1}'] = value  # Добавляем значения в словарь row_data

    data.append(row_data)
data

# %%
df = pd.DataFrame(data)
df = df.drop(columns=['input'])
df['Uptake'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Uptake']
df['Units'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Units']
df['Temp.'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Temp.']
df['Method'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Method']
df['Type'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Type']
df['Sequence'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Sequence']
df['Time, h'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Time, h']
df['Conc., uM'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['Conc., uM']
df['target'] = pd.read_csv('E:\\My_projects\\poseidon_Fluorescence (2).csv', index_col=0) ['target']
df

# %%
df.isna().sum()

# %%
df = df.dropna(subset = ['target']).drop(columns=['Uptake', 'Units'])
df.shape

# %%
df.target.isna().sum()

# %%
df = df.dropna(subset = ['Method', 'Type']).reset_index(drop=True)
df.shape

# %%
df

# %%
df.iloc[:, 4096: ]

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
knn=df.iloc[:, 4096: ].drop(columns=['Method',	'Type',	'Sequence'])
imputer = KNNImputer(n_neighbors = 5, weights = 'uniform')
knn_imputer=pd.DataFrame(imputer.fit_transform(knn))

# %%
knn_imputer

# %%
df['Temp.']=knn_imputer.iloc[:, 0]
df['Time, h']=knn_imputer.iloc[:, 1]
df['Conc., uM']=knn_imputer.iloc[:, 2]
df['target']=np.log(knn_imputer.iloc[:, 3] + 1e-6) 
df = df.drop(columns=['Type', 'Method', 'Sequence'])

# %%

df.rename(columns={col: i for i, col in enumerate(df.columns[:4096])}, inplace=True)

# %%
# @title Текст заголовка по умолчанию
import seaborn as sns
import matplotlib.pyplot as plt
def distribution(dt, col):
  #dt.drop(columns=['batch', 'patient'], inplace=True)
  x = str(col)
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (16, 8), sharex=False, sharey=False)
  fig.suptitle(x, fontsize=20)

  ax[0].title.set_text('distribution')
  variable = dt[x].fillna(dt[x].mean())
  sns.histplot(variable, kde=True, element='step', fill=True, alpha=.5, ax=ax[0])
  des = dt[x].describe()
  ax[0].axvline(des["25%"], ls='--')
  ax[0].axvline(des["mean"], ls='--')
  ax[0].axvline(des["75%"], ls='--')
  ax[0].grid(True)
  des = round(des, 2).apply(lambda x: str(x))
  box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"], "std: "+des["std"]))
  ax[0].text(0.25, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))

  ax[1].title.set_text('outliers')
  tmp_dtf = pd.DataFrame(dt[x])
  tmp_dtf.boxplot(column=x, ax=ax[1])
  plt.show()

# %%
def get_quantile_range_df(df, column_names):
    lower_quantiles = df[column_names].quantile(0.02)
    upper_quantiles = df[column_names].quantile(0.98)
    result = df
    for column in column_names:
        result = result[(result[column] >= lower_quantiles[column]) & (result[column] <= upper_quantiles[column])]
    return result

# %%
distribution(df, 'target')

# %%
df = get_quantile_range_df(df, ['target'])
df = df.reset_index(drop=True)
df.shape

# %%
distribution(df, 'target')

# %%
df.columns

# %%
df = df.rename(columns={'Temp.':'Temperature', 'Time, h': 'Time', 'Conc., uM': 'Conc'})

# %%
df

# %%
X = df.drop(columns=['target'])

y = df['target']

# %%
def adjusted_r2_score(r2, n, p):
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

# %%
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
def data_prep(X, y):
  X_scaled = X.copy()
  scalers = {}  # Словарь для хранения scaler'ов для каждого столбца

  for column in X.drop(columns=['Time', 'Conc', 'Temperature']).columns:
    scaler = MinMaxScaler()
    X_scaled[column] = scaler.fit_transform(X[[column]])

    scalers[column] = scaler
  if all(column in X.columns for column in ['Time', 'Conc', 'Temperature']):
    X_scaled['Time'] = X['Time'] 
    X_scaled['Conc'] = X['Conc'] 
    X_scaled['Temperature'] = X['Temperature'] 


  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



  return X_train, X_test, y_train, y_test


def adjusted_r2_score(r2, n, p):
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2


def cross_val_score(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True)
    r2_scores = []
    adj_r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        n = X_test.shape[0]
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)

    return np.mean(r2_scores), np.std(r2_scores), np.mean(adj_r2_scores), np.std(adj_r2_scores)


import lightgbm as lgb
import lightgbm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



X_train, X_valid, y_train, y_valid = data_prep(X, y)
# Создание модели LightGBM
model = lgb.LGBMRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# Метрики
print("Train:")
print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False)}")
print(f"R^2: {r2_score(y_train, y_train_pred)}")

print("\nValidation:")
print(f"RMSE: {mean_squared_error(y_valid, y_valid_pred, squared=False)}")
print(f"R^2: {r2_score(y_valid, y_valid_pred)}")

# Кросс-валидация
r2_mean= cross_val_score(model, X, y)
print("\nCross-Validation:")
print(f"Mean R2 Score: {r2_mean}")

# %%
feature_importances = model.feature_importances_
top_features = sorted(zip(feature_importances, X.columns), reverse=True)[:100]
top_features_names = [feature[1] for feature in top_features]
X = X[top_features_names]
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
high_cor = [column for column in upper.columns if any(upper[column] > 0.97)]
features = [i for i in top_features_names if i not in high_cor]
len(features)

# %%
X_train, X_valid, y_train, y_valid = data_prep(X[features], y)
# Создание модели LightGBM
model = lgb.LGBMRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# Метрики
print("Train:")
print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False)}")
print(f"R^2: {r2_score(y_train, y_train_pred)}")

print("\nValidation:")
print(f"RMSE: {mean_squared_error(y_valid, y_valid_pred, squared=False)}")
print(f"R^2: {r2_score(y_valid, y_valid_pred)}")

# Кросс-валидация
r2_mean= cross_val_score(model, X, y)
print("\nCross-Validation:")
print(f"Mean R2 Score: {r2_mean}")

# %%


# %%
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


print(features)
x = df.drop(columns=['target']).values
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(x)

# Создаем датафрейм для TSNE-результатов
tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne_x', 'tsne_y'])

plt.figure(figsize=(8, 8))
plt.scatter(tsne_df['tsne_x'], tsne_df['tsne_y'])

plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('TSNE Visualization')
plt.show()



# %%
combined_df = pd.concat([df, tsne_df], axis=1)
combined_df = combined_df.loc[(combined_df['tsne_x'] < 40) & (combined_df['tsne_x'] > -46)]

# %%
combined_df

# %%
X = combined_df.drop(columns=['tsne_x', 'tsne_y', 'target'])
y = combined_df['target']
distribution(combined_df, 'target')

# %%
X_train, X_valid, y_train, y_valid = data_prep(X, y)
# Создание модели LightGBM
model = lgb.LGBMRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# Метрики
print("Train:")
print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False)}")
print(f"R^2: {r2_score(y_train, y_train_pred)}")

print("\nValidation:")
print(f"RMSE: {mean_squared_error(y_valid, y_valid_pred, squared=False)}")
print(f"R^2: {r2_score(y_valid, y_valid_pred)}")

# Кросс-валидация
r2_mean= cross_val_score(model, X, y)
print("\nCross-Validation:")
print(f"Mean R2 Score: {r2_mean}")

# %%
feature_importances = model.feature_importances_
top_features = sorted(zip(feature_importances, X.columns), reverse=True)[:50]
top_features_names = [feature[1] for feature in top_features]
X = X[top_features_names]
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
high_cor = [column for column in upper.columns if any(upper[column] > 0.97)]
features = [i for i in top_features_names if i not in high_cor]

# %%
#y_log = np.log(y)
X_train, X_valid, y_train, y_valid = data_prep(X[features], y)
# Создание модели LightGBM
model = lgb.LGBMRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# Метрики
print("Train:")
print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False)}")
print(f"R^2: {r2_score(y_train, y_train_pred)}")

print("\nValidation:")
print(f"RMSE: {mean_squared_error(y_valid, y_valid_pred, squared=False)}")
print(f"R^2: {r2_score(y_valid, y_valid_pred)}")

# Кросс-валидация
r2_mean= cross_val_score(model, X[features], y)
print("\nCross-Validation:")
print(f"Mean R2 Score: {r2_mean}")

# %%
import optuna
#y_log = np.log(y)
def objective(trial,data=X[features],target=y):

    train_x, test_x, train_y, test_y = data_prep(data, target)
    param = {
        'metric': 'rmse',
        'random_state': 48,
        'n_estimators': 20000,  # Динамическое количество деревьев
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_int('max_depth', 3, 100),  # Изменено на int для большей гибкости
        'num_leaves': trial.suggest_int('num_leaves', 2, 1000),  # Минимальное значение изменено на 2
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100),
        'max_bin': trial.suggest_int('max_bin', 10, 255),  # Максимальное количество бинов
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-5, 10.0),  # Минимальный вес ребенка
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),  # Тип бустинга
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),  # Взвешивание положительных примеров
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0
    }
    model = lgb.LGBMRegressor(**param)

    model.fit(train_x,train_y,eval_set=[(test_x,test_y)])

    preds = model.predict(test_x)

    rmse = mean_squared_error(test_y, preds,squared=False)

    return rmse



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


params=study.best_params
params['random_state'] = 48
params['n_estimators'] = 20000
params['metric'] = 'rmse'
params['cat_smooth'] = params.pop('min_data_per_groups')

print('After hyperparams tuning')
X_train, X_valid, y_train, y_valid = data_prep(X[features], y)
print('Train:', len(X_train))
print('Valid:', len(X_valid), end='\n\n')

train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

data1 = lgb.Dataset(X, label=y)


print('Starting training...')

# train
gbm = lgb.train(params,
                train_data,
                num_boost_round=10000,
                valid_sets=[valid_data])
print()

# save model to file
print('Saving model...')
gbm.save_model('model.txt')

# predict
print('Starting predicting...')
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

# %%
from matplotlib.patches import Patch
y_train_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
r2_test = metrics.r2_score(y_valid, y_pred)
MAE_test = metrics.mean_absolute_error(y_valid, y_pred)
MSE_test = metrics.mean_squared_error(y_valid, y_pred)
RMSE_test = np.sqrt(metrics.mean_squared_error(y_valid, y_pred))
r2_train = metrics.r2_score(y_train, y_train_pred)
MAE_train = metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train = metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
n_test = len(y_valid)
p_test = X_valid.shape[1]
adj_r2_test = adjusted_r2_score(r2_test, n_test, p_test)

n_train = len(y_train)
p_train = X_train.shape[1]
adj_r2_train = adjusted_r2_score(r2_train, n_train, p_train)

print('r2_test:', r2_test)
print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('Adjusted r2_test:', adj_r2_test)

print('r2_train:', r2_train)
print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('Adjusted r2_train:', adj_r2_train)
print(X_train.shape)
real_patch = Patch(color='#DD7059', label='train values')
pred_patch = Patch(color='#569FC9', label='valid values')
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
f, ax = plt.subplots(figsize=(16, 12))
plt.scatter(y_train, y_train_pred, color='#DD7059', s=70)
plt.scatter(y_valid, y_pred, color='#569FC9',s=70)
plt.plot(y_valid, y_valid, color='gray')
plt.legend(handles=[real_patch, pred_patch, plt.Line2D([], [], color='gray', label='Ideal line')])
plt.title('LGBMRegressor')
plt.xlabel('true')
plt.ylabel('predict')
plt.savefig('lgbm_tuned.png')

# %%
lightgbm.plot_importance(gbm, max_num_features=50, figsize=(20,20), height=0.9)
plt.show()

feature_importance = gbm.feature_importances_

    # Нормализация значений важности признаков
normalized_importance = feature_importance / np.sum(feature_importance)
sorted_indices = np.argsort(normalized_importance)[::-1]  # Сортировка в порядке убывания

    # Ограничение количества значений для отображения (например, первые 10 значений)
top_n = 25
sorted_indices = sorted_indices[:top_n]

    # Сортировка значений и признаков
sorted_importance = normalized_importance[sorted_indices]
sorted_features = X_train.columns[sorted_indices]

    # Построение графика с отсортированными и ограниченными значениями важности признаков
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importance, y=sorted_features)
plt.xlabel('Normalized Importance')
plt.ylabel('Features')
plt.title('Top {} Feature Importance'.format(top_n))
plt.show()

# %%
X_train


