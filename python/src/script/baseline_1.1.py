# %% [markdown]
## 特徴量エンジニアリング
#=================================================


# %%
# # 特徴量エンジニアリング train_updated（rosters）,target1~4のラグ特徴量（1か月前）を使う

#%%
#Config
# =================================================

######################
# serial #
######################
serial_number = 2 #スプレッドシートAの番号


######################
# Data #
######################
input_path = '../input/mlb-player-digital-engagement-forecasting/' #フォルダ名適宜変更すること
file_path = "/tmp/work/src/exp/baseline_1.1.py" #ファイル名は適宜変更すること
file_name = os.path.splitext(os.path.basename(file_path))[0]



######################
# Dataset #
######################
# target_columns = 'TARGET'
# sub_index = 'SK_ID_CURR'

######################
# ハイパーパラメータの設定
######################
params={
	'boosting_type':'gbdt',
	'objective':'regression_l1',
	'metric':'mean_absolute_error',
	'learning_rate':0.05,
	'num_leaves':32,
	'subsample':0.7,
	'subsample_freq':1,
	'feature_fraction':0.8,
	'min_data_in_leaf':50,
	'min_sum_hessian_in_leaf':50,
	'n_estimators':1000,
	'random_state':123,
	'importance_type':'gain',
	}


#%%
#ライブラリ読み込み
# =================================================
import datetime as dt
import gc
import re
import os
import pickle
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
sns.set(font="IPAexGothic")
#!%matplotlib inline


#lightGBM
import lightgbm as lgb

#sckit-learn
# from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.metrics import mean_absolute_error

#表示桁数の設定
pd.options.display.float_format = '{:10.4f}'.format





#%%
# =================================================
# Utilities #
# =================================================

# 今の日時
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now

# %%
#メモリ削減関数
# =================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astypez(np.float64)
        else:
            pass
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100*((start_mem - end_mem) / start_mem):.2f}%')

    return df
                

#%%
#前処理の定義 カテゴリ変数をcategory型に
# =================================================
def data_pre00(df):
	for col in df.columns:
		if df[col].dtype == 'O':
			df[col] = df[col].astype('category')
	print('カテゴリ変数をcategory型に変換しました')
	df.info()
	return df

#%%
# 8-4:train_updated.csv専用の変換関数の作成
def unpack_json(json_str):
	return np.nan if pd.isna(json_str) else pd.read_json(json_str)

def extract_data(input_df, col='events', show=False):
	output_df = pd.DataFrame()
	for i in np.arange(len(input_df)):
		if show:print(f'\r{i + 1}/{len(input_df)}' ,end='')
		try:
			output_df = pd.concat([
				output_df,
				unpack_json(input_df[col].iloc[i])
			],axis=0,ignore_index=True)
		except:
			pass
	if show:print('')
	if show:print(output_df.shape)
	if show:display(output_df.head())
	return output_df

#%%

# 8-13:学習データと検証データの期間の設定
list_cv_month = [
	[['2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04',],['2021-05']],
	[['2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05',],['2021-06']],
	[['2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07',],['2021-07']],
	]

# 8-22:学習用関数の作成

def train_lgb(
	input_x,
	input_y,
	input_id,
	params,
	list_nfold=[0,1,2],
	mode_train='train',
	):

	#推論値を格納する変数の作成
	df_valid_pred=pd.DataFrame()
	#評価値を入れる変数の作成
	metrics=[]
	#重要度を格納するデータフレームの作成
	df_imp = pd.DataFrame()

	#output配下に現在のファイル名のフォルダを作成し、移動
	os.chdir('/tmp/work/src/output')
	if not os.path.isdir(file_name):
		os.makedirs(file_name)
		print(f'{file_name}フォルダ作成しました')
	os.chdir('/tmp/work/src/output/'+file_name)
	print(f'保存場所: {os.getcwd()}')
	
	#validation
	cv = []
	for month_tr, month_va in list_cv_month:
		cv.append([
			input_id.index[input_id['yearmonth'].isin(month_tr)],
			input_id.index[input_id['yearmonth'].isin(month_va) & (input_id['playerForTestSetAndFuturePreds'] == 1)],
		])

	
	#モデル学習(target/foldごとに学習)
	for nfold in list_nfold:
		for i, target in enumerate(['target1','target2','target3','target4']):
			print('-'*20,target, ',fold', nfold, '-'*20)
			#tainとvalidに分離
			idx_tr, idx_va = cv[nfold][0],cv[nfold][1]
			x_tr,y_tr,id_tr = input_x.loc[idx_tr,:],input_y.loc[idx_tr,target],input_id.loc[idx_tr,:]
			x_va,y_va,id_va = input_x.loc[idx_va,:],input_y.loc[idx_va,target],input_id.loc[idx_va,:]
			print(x_tr.shape,y_tr.shape,id_tr.shape)
			print(x_va.shape,y_va.shape,id_va.shape)

			#保存するモデルのファイル名
			filepath = f'model_lgb_{target}_fold{nfold}.h5'
			model = lgb.LGBMRegressor(**params)

			if mode_train == 'train':
				print('trainning start!')
				model.fit(
					x_tr,
					y_tr,
					eval_set=[(x_tr,y_tr),(x_va,y_va)],
					callbacks=[
						lgb.early_stopping(stopping_rounds=50, verbose=True),
						lgb.log_evaluation(100), 
						]
					)
				with open(filepath,'wb') as f:
					pickle.dump(model, f, protocol=4)
			else:
				print('model load.')
				with open(filepath, 'rb') as f:
					model = pickle.load(f)
				print('Done')
			
			#validの推論値取得
			y_va_pred = model.predict(x_va)
			tmp_pred = pd.concat([
				id_va,
				pd.DataFrame({'target':target,'nfold':nfold,'true':y_va,'pred':y_va_pred}),
				],axis=1)
			df_valid_pred = pd.concat([df_valid_pred,tmp_pred],axis=0,ignore_index=True)

			#評価値の算出
			metric_va = mean_absolute_error(y_va,y_va_pred)
			metrics.append([target,nfold,metric_va])

			#重要度の取得
			tmp_imp = pd.DataFrame({
				'col':x_tr.columns,
				'imp':model.feature_importances_,
				'target':'target1',
				'nfold':nfold,
			})
			df_imp = pd.concat([df_imp,tmp_imp],axis=0,ignore_index=True)
	
	print('-'*10,'result','-'*10)
	# 評価値
	df_metrics = pd.DataFrame(metrics,columns=['target','nfold','mae'])
	print(f'MCMAE:{df_metrics["mae"].mean():.4f}')

	#validの推論値
	df_valid_pred_all = pd.pivot_table(df_valid_pred, index=["engagementMetricsDate","playerId","date_playerId","date","yearmonth","playerForTestSetAndFuturePreds"], columns=["target", "nfold"], values=["true", "pred"], aggfunc=np.sum)
	df_valid_pred_all.columns = ["{}_fold{}_{}".format(j,k,i) for i,j,k in df_valid_pred_all.columns]
	df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)
	
	print('-'*20,'importance','-'*20)
	print(df_imp.groupby(['col'])['imp'].agg(['mean','std']).sort_values('mean',ascending=False)[:10])

	return df_valid_pred_all, df_metrics, df_imp

#%%
# 8-29:推論用データセット作成の関数

def makedataset_for_predict(input_test, input_prediction):
	test = input_test.copy()
	prediction = input_prediction.copy()

	#dateを日付型に変換
	prediction['date'] = pd.to_datetime(prediction['date'],format='%Y%m%d') 

	# engagementMetricsDateplayerIdのカラムを作成
	prediction['engagementMetricsDate'] = prediction['date_playerId'].apply(lambda x : x[:8])
	prediction['engagementMetricsDate'] = pd.to_datetime(prediction['engagementMetricsDate'],format='%Y%m%d')
	prediction['playerId'] = prediction['date_playerId'].apply(lambda x: int(x[9:]))

	#日付から曜日と年月を作成
	prediction['dayofweek'] = prediction['date'].dt.dayofweek
	prediction['yearmonth'] = prediction['date'].astype(str).apply(lambda x: x[:7])


	#テーブルの結合
	df_test = pd.merge(prediction,df_players,on='playerId',how='left')

	#説明関数の作成
	x_test = df_test[['playerId','dayofweek','birthCity', 'birthStateProvince','birthCountry', 'heightInches', 'weight', 'primaryPositionCode','primaryPositionName', 'playerForTestSetAndFuturePreds']]

	id_test = df_test[['engagementMetricsDate', 'playerId','date_playerId', 'date', 'yearmonth', 'playerForTestSetAndFuturePreds']]

	#カテゴリ変数をcategory型に変換
	data_pre00(x_test)

	return x_test, id_test

# %%
# 8-34:推論値処理の関数
def predict_lgb(
		input_test,
		input_id,
		list_nfold = [0,1,2],
		):
	df_test_pred = input_id.copy()

	for target in ['target1','target2','target3','target4']:
		for nfold in list_nfold:
			#modelのロード 
			with open(f'/tmp/work/src/output/baseline_1.0/model_lgb_{target}_fold{nfold}.h5','rb') as f:
				model = pickle.load(f)
				#推論
				pred = model.predict(input_test)
				#予測値の格納
				df_test_pred[f'{target}_{nfold}'] = pred
	#推論値の取得：各foldの平均値
	for target in ['target1','target2','target3','target4']:
		df_test_pred[target] = df_test_pred[df_test_pred.columns[df_test_pred.columns.str.contains(target)]].mean(axis=1)

	return df_test_pred



#%% ファイル

#ファイルの読み込み train_updated 1323行
# =================================================
# train = pd.read_csv(input_path+"train_updated.csv")
# print('train_updated:app_train')
# print(train.shape)
# display(train.head())

#ファイルの読み込み players
# =================================================
# df_players = pd.read_csv(input_path+"players.csv")
# print('players:app_train')
# print(df_players.shape)
# display(df_players.head())

# ====================
# 0 awards.csv
# ====================
# 3 players.csv
# ====================
# 4 seasons.csv
# ====================
# 5 teams.csv
# ====================
# 7 train_updated.csv
# ====================



# %% [markdown]
## 分析start!
#==========================================================
#%%


#%% ファイルの読み込み
#8-2:ファイルの読み込み
#ファイルの読み込み train_updated
# =================================================
# train = pd.read_csv(input_path+"train_updated.csv")
# print('train_updated:app_train')
# print(train.shape)
# display(train.head())

# #おそらくファイルサイズが大きくて読み込めないので、分割して読み込む
# # 読み込みたい大きなCSVファイルのパス
# input_file = input_path+'train_updated.csv'
# # 出力するファイルの名前のベース
# output_file_base = 'train_'
# # 一度に読み込む行数（ここでは500行ごとに分割）
# chunk_size = 500
# # チャンクごとにファイルを分割
# for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
# # 出力ファイル名
# 	output_file = f'{output_file_base}{i}.csv'
# 	# 分割したデータを保存
# 	chunk.to_csv(output_file, index=False)
# 	print(f'{output_file} が作成されました。')

#%%
# 8-2:ファイル読み込み続き、結合

# train_0 = pd.read_csv(input_path + 'train_0.csv')
# print(train_0.shape)
# display(train_0.head())

# train_1 = pd.read_csv(input_path + 'train_1.csv')
# print(train_1.shape)
# display(train_1.head())

# train_2 = pd.read_csv(input_path + 'train_2.csv')
# print(train_2.shape)
# display(train_2.head())

# train = pd.concat([train_0, train_1, train_2])

#%%
# 8-3:データの絞り込み
# train = train.loc[train['date']>=20200401, :].reset_index(drop=True)
# print(train.shape)

# train.to_csv('train_down.csv',index=None)

#%%ファイル読み込み
train = pd.read_csv(input_path + 'train_down.csv')
print(train.shape)
display(train.head())


# %%
# 8-5train_updated.csvから「nextDayPlayerEngagement」を取り出して表形式に変換
df_engagement = extract_data(train,col='nextDayPlayerEngagement',show=True)
# %%
# 8-6:結合キーであるdat_playIdの作成
df_engagement['date_playerId'] = df_engagement['engagementMetricsDate'].str.replace('-','') + '_'+ df_engagement['playerId'].astype(str)
df_engagement.tail()
# %%
# 8-7:日付から簡単な特徴量を作成
# 推論実施日のカラム作成（推論実施日＝推論対象日の前日）
df_engagement['date'] = pd.to_datetime(df_engagement['engagementMetricsDate'],format='%Y-%m-%d') + dt.timedelta(days=-1)

# 推論実施日から「曜日」と「年月」の特徴量作成
df_engagement['dayofweek'] = df_engagement['date'].dt.day_of_week
df_engagement['yearmonth'] = df_engagement['date'].astype(str).apply(lambda x: x[:7])
df_engagement.head()

# %%
# 8-8: players.csvの読み込み
df_players = pd.read_csv(input_path + "players.csv")
print(df_players.shape)
df_players.head()
df_players['playerForTestSetAndFuturePreds'] = np.where(df_players['playerForTestSetAndFuturePreds'] == True ,1, 0)

#%%[markdown]
# ↑↑これまでの処理
#==========================================================



#%%[markdown]
# 今回の実験（） 
#==========================================================



# %%
# 8-42 学習用データセット作成
# データ結合
df_train = pd.merge(df_engagement,df_players,on='playerId',how='left')
print(df_train.shape)



#説明変数と目的編素の作成
x_train = df_train[['playerId','dayofweek','birthCity', 'birthStateProvince','birthCountry', 'heightInches', 'weight', 'primaryPositionCode','primaryPositionName', 'playerForTestSetAndFuturePreds'] ]

y_train = df_train[['target1', 'target2', 'target3', 'target4']]

id_train = df_train[['engagementMetricsDate', 'playerId','date_playerId', 'date', 'yearmonth', 'playerForTestSetAndFuturePreds']]

#カテゴリに変換
data_pre00(x_train)

print(x_train.shape,y_train.shape,id_train.shape)
# %%8.4.3 モデル学習
# 8-43:モデル学習の実行

df_valid_pred, df_metrics, df_imp = train_lgb(
	x_train,
	y_train,
	id_train,
	params,
	list_nfold=[0,1,2],
	mode_train= 'train'
)

# %%
# 8-44:評価値の取得
print(f'MCMAE:{df_metrics["mae"].mean()}')
display(pd.pivot_table(df_metrics, index='nfold', columns='target', values='mae',aggfunc=np.mean,margins=True))

# %%
# 8-45:説明変数の重要度の確認
df_imp.groupby(['col'])['imp'].agg(['mean','std']).sort_values('mean',ascending=False)[:10]

# %% [markdown]
## モデル推論start!
# %%8.4.4 モデル推論

