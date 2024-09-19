# %% [markdown]
## ベースライン作成
#=================================================


# %%
# # ベースライン作成 rain_updated（nextDayPlayerEngagement）＋playersを使う
#%%
#!pip install mlb 

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
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

#表示桁数の設定
pd.options.display.float_format = '{:10.4f}'.format




#%%
#Config
# =================================================

######################
# serial #
######################
serial_number = 1 #スプレッドシートAの番号


######################
# Data #
######################
input_path = '../input/mlb-player-digital-engagement-forecasting/' #フォルダ名適宜変更すること
file_path = "/tmp/work/src/exp/baseline_1.2.py" #ファイル名は適宜変更すること
file_name = os.path.splitext(os.path.basename(file_path))[0]


######################
# Dataset #
######################
# target_columns = 'TARGET'
# sub_index = 'SK_ID_CURR'

######################
# ハイパーパラメータの設定
######################
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'learning_rate': 0.05,
#     'num_leaves': 32,
#     'n_estimators':100000,
#     'random_state': 123,
#     'importance_type': 'gain',
# }


# =================================================
# Utilities #
# =================================================

# 今の日時
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now



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
#ファイルの確認
# =================================================
# for dirname, _, filenames in os.walk(input_path):
#     for i, datafilename in enumerate(filenames):
#         # print(os.path.join(dirname,filename))
#         print('='*20)
#         print(i,datafilename)


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

train_0 = pd.read_csv(input_path + 'train_0.csv')
print(train_0.shape)
display(train_0.head())

train_1 = pd.read_csv(input_path + 'train_1.csv')
print(train_1.shape)
display(train_1.head())

train_2 = pd.read_csv(input_path + 'train_2.csv')
print(train_2.shape)
display(train_2.head())

# %%
# 8-2:ファイル読み込み続き、結合
train = pd.concat([train_0, train_1, train_2])
train.shape
# %%
# 8-3:データの絞り込み
train = train.loc[train['date']>=20200401, :].reset_index(drop=True)
print(train.shape)

# %%
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
print(df_players['playerId'].agg('nunique'))
df_players.head()
# %%
df_players.columns
# %%
# 8-9 評価対象の人数の確認
df_players['playerForTestSetAndFuturePreds'] = np.where(df_players['playerForTestSetAndFuturePreds'] == True ,1, 0)
print(df_players['playerForTestSetAndFuturePreds'].sum())
print(df_players['playerForTestSetAndFuturePreds'].mean())
# %% 8.3.3データセット作成
# 8-10 テーブル結合
df_train = pd.merge(df_engagement,df_players,on='playerId',how='left')
print(df_train.shape)
df_train.head()

# %%
df_train.columns
# %%
# 8-11 学習用データセット作成
x_train = df_train[['playerId','dayofweek','birthCity', 'birthStateProvince','birthCountry', 'heightInches', 'weight', 'primaryPositionCode','primaryPositionName', 'playerForTestSetAndFuturePreds']]
y_train = df_train[['target1', 'target2', 'target3', 'target4']]

id_train = df_train[['engagementMetricsDate', 'playerId','date_playerId', 'date', 'yearmonth', 'playerForTestSetAndFuturePreds']]

print(x_train.shape,y_train.shape,id_train.shape)
x_train.head()
# %%
data_pre00(x_train)
# %%
x_train.info()
# %% 8.3.4 バリデーション設計
# 8-13:学習データと検証データの期間の設定
list_cv_month = [
	[['2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04',],['2021-05']],
	[['2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05',],['2021-06']],
	[['2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07',],['2021-07']],
	]


# %%
# 8-14:学習データと検証データのindexリストの作成
cv = []
for month_tr, month_va in list_cv_month:
	cv.append([
		id_train.index[id_train['yearmonth'].isin(month_tr)],
		id_train.index[id_train['yearmonth'].isin(month_va) & (id_train['playerForTestSetAndFuturePreds'] == 1)],
	])
#fold0のindexのリスト
cv[0]
# %% 8.3.5モデル学習
# 8-15:学習データと検証データに分離
target = 'target1'
nfold = 0

#trainとvalidのindex取得
idx_tr, idx_va = cv[nfold][0],cv[nfold][1]

#trainデータとvalidデータに分離
x_tr,y_tr,id_tr = x_train.loc[idx_tr,:],y_train.loc[idx_tr,target],id_train.loc[idx_tr,:]
x_va,y_va,id_va = x_train.loc[idx_va,:],y_train.loc[idx_va,target],id_train.loc[idx_va,:]
print(x_tr.shape,y_tr.shape,id_tr.shape)
print(x_va.shape,y_va.shape,id_va.shape)

# %%
# 8-16:モデル学習
#ハイパーパラメータの設定
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

#モデル学習
model = lgb.LGBMRegressor(**params)
model.fit(
	x_tr,
	y_tr,
	eval_set=[(x_tr,y_tr),(x_va,y_va)],
	callbacks=[
		lgb.early_stopping(stopping_rounds=50, verbose=True),
		lgb.log_evaluation(100), 
		]
	)

#モデル保存
with open('model_lgb_target1_fold0.h5','wb') as f:
	pickle.dump(model, f, protocol=4)
# %%
# 8-17 モデルの評価
#validデータの推論値取得
y_va_pred = model.predict(x_va)

#全target/foldの推論値を格納する変数の作成
df_valid_pred =pd.DataFrame()

#推論を格納
tmp_pred = pd.concat([
	id_va,
	pd.DataFrame({'target':target,'nfold':0,'true':y_va,'pred':y_va_pred}),
],axis=1)
df_valid_pred = pd.concat([df_valid_pred,tmp_pred],axis=0,ignore_index=True)

#全target/foldの評価値を格納する変数の作成
metrics = []

#評価値の算出
metric_va = mean_absolute_error(y_va,y_va_pred)

#評価値を格納
metrics.append([target,nfold,metric_va])
metrics
# %%
# 8-18:説明変数の重要度取得
#重要度の取得
tmp_imp = pd.DataFrame({
	'col':x_tr.columns,
	'imp':model.feature_importances_,
	'taeget':'target1',
	'nfold':nfold,
})
#確認（重要度の上位10個）
display(tmp_imp.sort_values('imp',ascending=False))
#全target/foldの重要度を格納するデータフレームの作成
df_imp = pd.DataFrame()

#imp_foldをdm_impに結合
df_imp = pd.concat([df_imp,tmp_imp],axis=0,ignore_index=True)
# %%
# 8-19:モデルの評価
# リスト型をデータフレームに変換
df_metrics = pd.DataFrame(metrics,columns=['target','nfold','mae'])
display(df_metrics.head())

# 評価値
print(f'MCMAE:{df_metrics["mae"].mean():.4f}')

display(pd.pivot_table(df_metrics,index='nfold',columns='target',values='mae',aggfunc=np.mean,margins=True))
#%% 
display(df_valid_pred.head())
print(df_valid_pred.columns)
# %%
# 8-20:検証データの推論値の形式変換（全target/foldのサマリ）
df_valid_pred_all = pd.pivot_table(df_valid_pred, index=["engagementMetricsDate","playerId","date_playerId","date","yearmonth","playerForTestSetAndFuturePreds"], columns=["target",  "nfold"], values=["true", "pred"], aggfunc=np.sum)
df_valid_pred_all.columns = ["{}_fold{}_{}".format(j,k,i) for i,j,k in df_valid_pred_all.columns]
df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)
df_valid_pred_all.head()
# %%
df_valid_pred_all.shape
# %%
# 8-21:説明変数の重要度の取得（全target/foldのサマリ）
df_imp.groupby(['col'])['imp'].agg(['mean','std']).sort_values('mean',ascending=False)



# %%
# 8-22:学習用関数の作成

file_path = "/tmp/work/src/script/baseline_1.0.py" #ファイル名は適宜変更すること
file_name = os.path.splitext(os.path.basename(file_path))[0] 


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
	print(df_imp.groupby(['col'])['imp'].agg(['mean','std']).sort_values('mean',ascending=False))

	return df_valid_pred_all, df_metrics, df_imp




#%%
# 8-23:モデル学習
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


df_valid_pred, df_metrics, df_imp = train_lgb(
	x_train,
	y_train,
	id_train,
	params,
	list_nfold=[0,1,2],
	mode_train= 'train'
)

#%%
# 8-24:評価値の確認
print(f'MCMAE:{df_metrics["mae"].mean()}')
display(pd.pivot_table(df_metrics, index='nfold', columns='target', values='mae',aggfunc=np.mean,margins=True))

# %%
# 8-25:説明変数の重要度の取得
df_imp.groupby(['col'])['imp'].agg(['mean','std']).sort_values('mean',ascending=False)


# %% [markdown]
## モデル推論start!


# %%パート1推論用データセットの作成
# 8-26:データフォーマットの確認(サブミット時にはコメントアウト)


# env = MLB.make_env()
# iter_test = env.iter_test()

# for(test_df, prediction_df) in iter_test:
#     display(test_df.head())
#     display(prediction_df.head())
#     break

#%%
#8-27:推論時に受け取るデータのフォーマット確認②（サブミット時はコメントアウト）
# os.chdir('/tmp/work/src/exp')

# test_df = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/example_test.csv")
# display(test_df.head())

# prediction_df = pd.read_csv("../input/mlb-player-digital-engagement-forecasting/example_sample_submission.csv")
# display(prediction_df.head())

#%%
# 8-28:推論時に受け取るデータの疑似生成（2021/4/26分）
test_df = train.loc[train['date'] == 20210426,:]
display(test_df.head())

#prediction_dfの疑似生成（4/26に受け取るデータを想定）
prediction_df = df_engagement.loc[df_engagement['date'] == '2021-04-26',['date','date_playerId']].reset_index(drop=True)
prediction_df['date'] = prediction_df['date'].apply(lambda x : int(str(x).replace('-','')[:8]))
for col in ['target1', 'target2', 'target3', 'target4']:
	prediction_df[col] = 0
display(prediction_df.head())

#%%
# 8-29:推論用データセット作成の関数

def makedataset_for_predict(input_test, input_prediction):
	test = input_test.copy()
	prediction = input_prediction.copy()

	#dateを日付型に変換
	prediction['date'] = pd.to_datetime(prediction['date'],format='%Y%m%d') 

	# 推論対象日(engagementMetricsDate)と選手ID(playerId)のカラムを作成
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
# 8-30:推論用データセットの作成の実行
x_test, id_test = makedataset_for_predict(test_df, prediction_df)
display(x_test.head())
display(id_test.head())


# %%パート2：モデル推論
# 8-31:モデルの読み込み
with open('/tmp/work/src/output/baseline_1.0/model_lgb_target1_fold0.h5','rb') as f:
	model = pickle.load(f)


# %%
# 8-32:モデルを用いた推論
pred = model.predict(x_test)

df_test_pred = id_test.copy()
df_test_pred['target1_fold0'] = pred
# %%
# 8-33:推論値の計算
#target1の推論値：各foldの平均値
df_test_pred['target1'] = df_test_pred[df_test_pred.columns[df_test_pred.columns.str.contains('target1')]].mean(axis=1)

print(df_test_pred.shape)
df_test_pred.head()
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


# %%
# 8-35:モデル推論の実行
df_test_pred = predict_lgb(x_test,id_test)
df_test_pred.head()
# %%パート３：提出用フォーマットへの変換
# 8-36:提出用フォーマットへの変換
df_submit = df_test_pred[['date_playerId','target1', 'target2', 'target3', 'target4']]
df_submit.head()







# %%
# 8-37:推論処理の実行（まとめ）

import mlb # type:ignore

env = mlb.make_env()
iter_test = env.iter_test()

for(test_df, prediction_df) in iter_test:
	test = test_df.copy()
	prediction = prediction_df.copy()
	prediction = prediction.reset_index(drop=False)

	print('date',prediction['date'][0])
	#データセット作成
	x_test, id_test = makedataset_for_predict(test, prediction)
	#推論処理
	df_test_pred = predict_lgb(x_test,id_test)

	#提出データの作成
	df_submit = df_test_pred[['date_playerId','target1', 'target2', 'target3', 'target4']]
	#後処理：欠損値埋め、0-100の範囲以外のデータのクリッピング
	for i,col in enumerate(['target1', 'target2', 'target3', 'target4']):
		df_submit[col] = df_submit[col].fillna(0.)
		df_submit[col] = df_submit[col].clip(0,100)

	#予測データの提出
	env.predict(df_submit)
print('Done')
		



