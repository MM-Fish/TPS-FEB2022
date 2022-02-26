import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
from util import Logger, Util
from base import Feature, get_arguments, generate_features
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from scipy.sparse.csgraph import connected_components
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import warnings

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)

RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
EDA_DIR_NAME = yml['SETTING']['EDA_DIR_NAME']  # EDAに関する情報を格納場所
Feature.dir = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
feature_memo_path = Feature.dir + '_features_memo.csv'


def create_features(self):
    self.train = train.iloc[:, 1:-1].copy()
    self.test = test.iloc[:, 1:].copy()
    create_memo('all_raw_data', '全初期データ')


# ##PCA
class pca_output(Feature):
    def create_features(self):
        out_dir_name = EDA_DIR_NAME + 'pca' + '/'
        my_makedirs(out_dir_name)
        logger = Logger(out_dir_name)

        model_pca = PCA()
        feature = model_pca.fit_transform(all_data.drop(['row_id'], axis=1).select_dtypes(exclude='object'))
        accum_contribution_rate = np.cumsum(model_pca.explained_variance_ratio_)

        # グラフ化
        # 散布図と寄与率
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(feature[:train_size, 0], feature[:train_size, 1], alpha=0.8, hue=target, data=train, ax=axes[0])
        sns.lineplot([n for n in range(1, len(accum_contribution_rate)+1)], accum_contribution_rate, markers=True, ax=axes[1])
        plt.savefig(out_dir_name + f'pca{suffix}.png', dpi=300, bbox_inches="tight")
        plt.close()

        logger.info('explained variance ratio: {}'.format(accum_contribution_rate))
        self.train = pd.DataFrame(feature).iloc[:train_size, accum_contribution_rate <= 0.8].add_prefix('pca')
        self.test = pd.DataFrame(feature).iloc[train_size:, accum_contribution_rate <= 0.8].add_prefix('pca')
         
# # ##t-SNE
# class tsne_output(Feature):
#     def create_features(self):
#         model_tsne = TSNE(n_components=2, perplexity=10)
#         feature = model_tsne.fit_transform(all_data.drop(['row_id'], axis=1).select_dtypes(exclude='object'))

#         # グラフ化
#         fig, ax = plt.subplots(figsize=(6, 6))
#         sns.scatterplot(feature[:train_size, 0], feature[:train_size, 1], alpha=0.8, hue=target, data=train, ax=ax)
#         plt.savefig(EDA_DIR_NAME + f'tsne{suffix}.png', dpi=300, bbox_inches="tight")
#         plt.close()

#         self.train, self.test = pd.DataFrame(feature).iloc[:train_size, :].add_prefix('tsne'), pd.DataFrame(feature).iloc[train_size:, :].add_prefix('tsne')

# # ##UMAP
# class umap_output(Feature):
#     def create_features(self):
#         model_umap = umap.UMAP(n_components=2, n_neighbors=10)
#         feature = model_umap.fit_transform(all_data.drop(['row_id'], axis=1).select_dtypes(exclude='object'))

#         # グラフ化
#         fig, ax = plt.subplots(figsize=(6, 6))
#         sns.scatterplot(feature[:train_size, 0], feature[:train_size, 1], alpha=0.8, hue=target, data=train, ax=ax)
#         plt.savefig(EDA_DIR_NAME + f'umap{suffix}.png', dpi=300, bbox_inches="tight")
#         plt.close()

#         self.train, self.test = pd.DataFrame(feature).iloc[:train_size, :].add_prefix('umap'), pd.DataFrame(feature).iloc[train_size:, :].add_prefix('umap')

# 特徴量メモcsvファイル作成
def create_memo(col_name, desc):

    file_path = Feature.dir + '/_features_memo.csv'
    if not os.path.isfile(file_path):
        with open(file_path,"w") as f:
            writer = csv.writer(f)
            writer.writerow([col_name, desc])

    with open(file_path, 'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])

def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    now = datetime.datetime.now()
    suffix = now.strftime("_%m%d_%H%M")
    target = 'target'

    # CSVのヘッダーを書き込み
    create_memo('特徴量', 'メモ')

    args = get_arguments()
    train = pd.read_csv(RAW_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DIR_NAME + 'test.csv')
    all_data = pd.concat([train, test])
    train_size = len(train)
    test_size = len(test)

    # trainにおける正解データ（作図に使用）
    encoded_targets = train[target].map(lambda x: yml['SETTING']['TARGET_ENCODING'][x]).values

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)