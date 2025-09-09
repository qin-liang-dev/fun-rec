# 数据准备
from typing import Dict, List

from pandas import DataFrame
import pandas as pd
import numpy as np


def init_user_date() -> Dict[str, Dict[str, int]]:
    """
    初始化用户数据字典

    Returns:
        Dict[str, Dict[str, int]]: 返回一个嵌套字典，外层键为项目名称，
        内层键为用户名称，值为用户对项目的评分（整数）
    """
    return {
        'item1': {'user1': 5, 'user2': 3, 'user3': 4, 'user4': 3, 'user5': 1},
        'item2': {'user1': 3, 'user2': 1, 'user3': 3, 'user4': 3, 'user5': 5},
        'item3': {'user1': 4, 'user2': 2, 'user3': 4, 'user4': 1, 'user5': 5},
        'item4': {'user1': 4, 'user2': 3, 'user3': 3, 'user4': 5, 'user5': 2},
        'item5': {'user2': 3, 'user3': 5, 'user4': 4, 'user5': 1},
    }


def calc_similarity_matrix(item_data: Dict[str, Dict[str, int]]) -> DataFrame:
    """
    计算物品相似度矩阵

    该函数通过计算物品之间的皮尔逊相关系数来构建相似度矩阵。
    对于每一对物品，找出同时评价过这两个物品的用户，然后基于这些用户的评分计算相关系数。

    Args:
        item_data (Dict[str, Dict[str, int]]): 物品数据字典，格式为 {物品ID: {用户ID: 评分}}

    Returns:
        DataFrame: 物品相似度矩阵，行列索引均为物品ID，对角线为1（自己与自己的相似度）
    """
    item_ids: List[str] = list(item_data.keys())
    similarity_matrix = pd.DataFrame(
        np.identity(len(item_data)),
        index=item_ids,
        columns=item_ids,
    )
    # 遍历每条物品-用户评分数据
    for i1, users1 in item_data.items():
        for i2, users2 in item_data.items():
            if i1 == i2:
                continue
            vec1, vec2 = [], []
            # 收集同时评价过两个物品的用户评分数据
            for user, rating1 in users1.items():
                rating2 = users2.get(user, -1)
                if rating2 == -1:
                    continue
                vec1.append(rating1)
                vec2.append(rating2)
            # 计算两个物品评分向量的皮尔逊相关系数
            similarity_matrix[i1][i2] = np.corrcoef(vec1, vec2)[0][1]
    # print(similarity_matrix)
    return similarity_matrix


class ItemCf:
    def __init__(self, item_data: Dict[str, Dict[str, int]]):
        self.item_data = item_data
        # 基于皮尔逊相关系数，计算物品相似性矩阵
        self.similarity_matrix = calc_similarity_matrix(item_data)

    def find_similarity_item(self, target_user: str, target_item: str, num: int) -> List[str]:
        """
        查找与目标物品最相似的物品中，目标用户评分过的前N个物品
        Args:
            target_user (str): 目标用户的标识符
            target_item (str): 目标物品的标识符
            num (int): 需要查找的相似物品数量
        Returns:
            无返回值，直接打印结果
        """
        sim_items = []
        sim_items_list = self.similarity_matrix[target_item].sort_values(ascending=False).index.tolist()
        for item in sim_items_list:
            # 如果target_user对物品item评分过
            if target_user in self.item_data[item]:
                sim_items.append(item)
            if len(sim_items) == num:
                break
        print(f'与物品{target_item}最相似的{num}个物品为：{sim_items}')
        return sim_items

    def pred_score(self, target_user: str, target_item: str, num: int) -> int:
        target_user_mean_rating = np.mean(list(self.item_data[target_item].values()))
        weighted_scores = 0.
        corr_values_sum = 0.

        sim_items = self.find_similarity_item(target_user, target_item, num)
        for item in sim_items:
            corr_value = self.similarity_matrix[target_item][item]
            user_mean_rating = np.mean(list(self.item_data[item].values()))

            weighted_scores += corr_value * (self.item_data[item][target_user] - user_mean_rating)
            corr_values_sum += corr_value

        target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
        # print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')
        return target_item_pred
