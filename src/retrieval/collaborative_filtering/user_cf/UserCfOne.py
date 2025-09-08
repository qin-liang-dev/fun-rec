from typing import Dict, List
import pandas as pd

import numpy as np
from pandas import DataFrame


# 数据准备
def init_user_date() -> Dict:
    return {
        'user1': {'item1': 5, 'item2': 3, 'item3': 4, 'item4': 4},
        'user2': {'item1': 3, 'item2': 1, 'item3': 2, 'item4': 3, 'item5': 3},
        'user3': {'item1': 4, 'item2': 3, 'item3': 4, 'item4': 3, 'item5': 5},
        'user4': {'item1': 3, 'item2': 3, 'item3': 1, 'item4': 5, 'item5': 4},
        'user5': {'item1': 1, 'item2': 5, 'item3': 5, 'item4': 2, 'item5': 1},
    }


# 基于皮尔逊相关系数，计算用户相似性矩阵
def calc_similarity_matrix(user_data: Dict[str, Dict[str, int]]) -> DataFrame:
    user_ids: List = list(user_data.keys())
    similarity_matrix: DataFrame = pd.DataFrame(
        np.identity(len(user_data)),
        index=user_ids,
        columns=user_ids,
    )

    # 遍历每条用户-物品评分数据
    for u1, items1 in user_data.items():
        for u2, items2 in user_data.items():
            if u1 == u2:
                continue
            vec1, vec2 = [], []
            for item, rating1 in items1.items():
                rating2 = items2.get(item, -1)
                if rating2 == -1:
                    continue
                vec1.append(rating1)
                vec2.append(rating2)
            # 计算不同用户之间的皮尔逊相关系数
            similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]
    return similarity_matrix


def training_model():
    import funrec
    from funrec.utils import build_metrics_table

    # 加载配置
    config = funrec.load_config('user_cf')

    # 加载数据
    train_data, test_data = funrec.load_data(config.data)

    # 准备特征
    feature_columns, processed_data = funrec.prepare_features(config.features, train_data, test_data)

    # 训练模型
    models = funrec.train_model(config.training, feature_columns, processed_data)

    # 评估模型
    metrics = funrec.evaluate_model(models, processed_data, config.evaluation, feature_columns)

    print(build_metrics_table(metrics))


class UserCf:
    def __init__(self, user_data: Dict[str, Dict[str, int]]):
        self.user_data = user_data
        # 基于皮尔逊相关系数，计算用户相似性矩阵
        self.similarity_matrix = calc_similarity_matrix(user_data)

    def find_similarity_users(self, target_user: str, top_n: int) -> List[str]:
        """ 计算与目标用户最相似的前top_n个用户
        Args:
            target_user(str): 目标用户
            top_n(int): 最相似用户数量
        Returns:
            List: 最相似用户列表
        """
        # 由于最相似的用户为自己，去除本身
        sim_users: List[str] = self.similarity_matrix[target_user].sort_values(ascending=False)[1:top_n + 1].index.tolist()
        return sim_users

    def pred_score(self, target_user: str, top_n: int, target_item: str) -> int:
        """ 基于前top_n个最相似的用户,预测用户target_user对物品target_item的评分
        Args:
            target_user(str): 目标用户
            top_n(int): 最相似用户数量
            target_item(str): 目标物品
        Returns:
            int: 预测评分
        """
        weighted_scores = 0.
        corr_values_sum = 0.
        sim_users: List = self.find_similarity_users(target_user, top_n)
        # 基于皮尔逊相关系数预测用户评分
        for user in sim_users:
            corr_value = self.similarity_matrix[target_user][user]
            user_mean_rating = np.mean(list(self.user_data[user].values()))
            weighted_scores += corr_value * (self.user_data[user][target_item] - user_mean_rating)
            corr_values_sum += corr_value
        target_user_mean_rating = np.mean(list(self.user_data[target_user].values()))
        target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
        return target_item_pred


def test_find_similarity_users():
    cf = UserCf(init_user_date())
    print(f'用户相似性矩阵: \n{cf.similarity_matrix}')

    target_user = 'user1'
    top_n = 2
    sim_users = cf.find_similarity_users(target_user, top_n)
    print(type(sim_users[0]))
    print(f'与用户{target_user}最相似的{top_n}个用户为：{sim_users}')


def test_pred_score():
    cf = UserCf(init_user_date())
    target_item = 'item5'
    target_user = 'user1'
    top_n = 2
    target_item_pred = cf.pred_score(target_user, top_n, target_item)
    print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred:.4f}')


if __name__ == '__main__':
    test_find_similarity_users()
    # test_pred_score()
    # training_model()
