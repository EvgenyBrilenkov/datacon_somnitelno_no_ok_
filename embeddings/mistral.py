import optuna
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

with open("/mnt/tank/scratch/igolovkin/train_result.json", "r") as f:
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

# Создаем DataFrame
train = pd.DataFrame(data)
train = train.drop(columns=['input'])
train['Yield'] = pd.read_csv('/mnt/tank/scratch/igolovkin/train_data.csv')['Yield']
print(train)


with open("/mnt/tank/scratch/igolovkin/result_test.json", "r") as f:
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

test = pd.DataFrame(data)
test= test.drop(columns=['input'])
test['Yield'] = pd.read_csv('/mnt/tank/scratch/igolovkin/test_data.csv')['Yield']


with open("/mnt/tank/scratch/igolovkin/result_validate.json", "r") as f:
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

val = pd.DataFrame(data)
val = val.drop(columns=['input'])
val['Yield'] = pd.read_csv('/mnt/tank/scratch/igolovkin/validate_data.csv')['Yield']




target=train['Yield']


X_train = train.drop(columns='Yield')
y_train = train['Yield']
X_val = val.drop(columns='Yield')
y_val = val['Yield']
X_test = test.drop(columns='Yield')
y_test = test['Yield']



def objective(trial, train = train, val = val):
    
    X_train = train.drop(columns='Yield')
    X_val = val.drop(columns='Yield')
    y_train = train['Yield']
    y_val = val['Yield']
    param = {
        'metric': 'rmse', 
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100),
	"device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0
    }
    model = LGBMRegressor(**param)  
    
    model.fit(X_train,y_train,eval_set=[(X_val,y_val)])
    
    preds = model.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds,squared=False)
    
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

preds = np.zeros(test.shape[0])
kf = KFold(n_splits=5,random_state=48,shuffle=True)
rmse=[]  # list contains rmse for each fold
n=0
from sklearn.metrics import r2_score

# Список для хранения R2 для каждой части в кросс-валидации
r2_scores = []
columns = train.drop(columns='Yield').columns
for trn_idx, test_idx in kf.split(train[columns],train['Yield']):
    X_tr,X_val=train[columns].iloc[trn_idx],train[columns].iloc[test_idx]
    y_tr,y_val=train['Yield'].iloc[trn_idx],train['Yield'].iloc[test_idx]
    model = LGBMRegressor(**params)
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)])
    preds+=model.predict(test[columns])/kf.n_splits
    rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
    print(n+1,rmse[n])
    r2 = r2_score(y_val, model.predict(X_val))
    r2_scores.append(r2)
    print(n+1, 'R2:', r2)
    n+=1



print('Mean rmse', np.mean(rmse))
print('Mean r2', np.mean(r2))

model = LGBMRegressor(**params)
model.fit(train[columns], train['Yield'])

# Предсказываем значения для датасета test
predictions = model.predict(test[columns])

# Рассчитываем коэффициент детерминации R2 для тестового набора данных
r2_test = r2_score(test['Yield'], predictions)
print('R2 для тестового набора данных:', r2_test)

# Рассчитываем RMSE для тестового набора данных
rmse_test = np.sqrt(mean_squared_error(test['Yield'], predictions))
print('RMSE для тестового набора данных:', rmse_test)



def adjusted_r2_score(r2, n, p):
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2
from matplotlib.patches import Patch
y_train_pred = model.predict(X_train)
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