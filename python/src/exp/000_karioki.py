
#説明変数と目的編素の作成
x_train = df_train[['playerId','dayofweek','birthCity', 'birthStateProvince','birthCountry', 'heightInches', 'weight', 'primaryPositionCode','primaryPositionName', 'playerForTestSetAndFuturePreds'] + col_rosters + col_agg_target]

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


# %%
# 8-30:推論用データセットの作成の実行
x_test, id_test = makedataset_for_predict(test_df, prediction_df)
display(x_test.head())
display(id_test.head())


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

# import mlb # type:ignore

# env = mlb.make_env()
# iter_test = env.iter_test()

# for(test_df, prediction_df) in iter_test:
# 	test = test_df.copy()
# 	prediction = prediction_df.copy()
# 	prediction = prediction.reset_index(drop=False)

# 	print('date',prediction['date'][0])
# 	#データセット作成
# 	x_test, id_test = makedataset_for_predict(test, prediction)
# 	#推論処理
# 	df_test_pred = predict_lgb(x_test,id_test)

# 	#提出データの作成
# 	df_submit = df_test_pred[['date_playerId','target1', 'target2', 'target3', 'target4']]
# 	#後処理：欠損値埋め、0-100の範囲以外のデータのクリッピング
# 	for i,col in enumerate(['target1', 'target2', 'target3', 'target4']):
# 		df_submit[col] = df_submit[col].fillna(0.)
# 		df_submit[col] = df_submit[col].clip(0,100)

# 	#予測データの提出
# 	env.predict(df_submit)
# print('Done')
		



