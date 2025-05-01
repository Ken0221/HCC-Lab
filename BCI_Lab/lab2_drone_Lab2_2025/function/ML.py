import numpy as np
import joblib
from mne.decoding import CSP as CSP_
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
# CSP(共通空間模式):
class CSP(object):
  def __init__(self, csp_clf_path):
    self.clf = None                       # CSP用 分類器
    self.best_n_com = 0                   # 最佳特徵空間數
    self.csp_clf_path = csp_clf_path      # csp 已訓練完成的模型
  
  # 取得最佳超參數
  def get_best_feature(self, X, y):
    best_score = 0
    best_n_components = None
    scores_dict = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    for n_com in [2, 4, 6, 8]:
      clf = Pipeline([('CSP', CSP_(n_components=n_com, reg=0.05)), 
                ('LDA', LinearDiscriminantAnalysis())])
      scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
      mean_score = np.mean(scores)
      scores_dict[n_com] = mean_score
      # print(f"CSP components: {n_com}, Accuracy: {mean_score*100:.2f} %")
      if mean_score > best_score:
        best_score = mean_score
        best_n_components = n_com
    # 更新
    self.best_n_com = best_n_components
    print(f"  CSP(n={best_n_components}) Accuracy: {scores_dict[best_n_components]*100:.2f} %")
  # 透過CSP 對所有資料做 train
  def csp_all_fit(self, X, y):
    self.clf = Pipeline([ ('CSP', CSP_(n_components=self.best_n_com, reg=0.05)), 
                          ('LDA', LinearDiscriminantAnalysis())])
    self.clf.fit(X, y)
  # 輸出模型:
  def output_model(self):
    joblib.dump(self.clf, self.csp_clf_path)

# 由於 CSP 是設計給二分類的，故額外多寫以下部分去進行多類別分類
# 策略:將左右合成成一個label(特徵較相似)、rest 為一個 label
def ML(PS,X,y):
  csp_clf_path1 = f'EEG_data/{PS.user_name}_model1.joblib'
  csp_clf_path2 = f'EEG_data/{PS.user_name}_model2.joblib'
  # 前處理: 去掉 P300 成份
  X = X[:,:,500:1000]
  # 0:rest / 1:左 / 2:右 
  # 把左和右 整合成一個 label
  y_1 = [1 if val == 2 else val for val in y]
  # 移除等於 0的特徵
  y_2 = np.array(y)
  X_2 = X[y_2!=0,:,:]
  y_2 = y_2[y_2!=0]
  # 先訓練一個 判斷是否有左右的 model
  print("判斷是否動作:")
  csp1 = CSP(csp_clf_path1)
  csp1.get_best_feature(X, y_1)
  csp1.csp_all_fit(X, y_1)
  csp1.output_model()
  # 在訓練一個 判斷左右的 model
  print("判斷左右:")
  csp2 = CSP(csp_clf_path2)
  csp2.get_best_feature(X_2, y_2)
  csp2.csp_all_fit(X_2, y_2)
  csp2.output_model()