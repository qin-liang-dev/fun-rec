import sys

import funrec


def calculate_swing_similarity(item_users, user_items, user_weights, item_i, item_j, alpha=1.0):
    """
    计算两个物品之间的Swing Score相似度
    参考funrec.models.swing.Swing._calculate_swing_similarity_optimized()的核心逻辑
    """
    # 找到同时与物品i和j交互的用户（共同用户）
    common_users = item_users[item_i].intersection(item_users[item_j])

    if len(common_users) < 2:
        return 0.0  # 至少需要2个共同用户才能计算Swing score

    swing_score = 0.0
    common_users_list = list(common_users)

    # 计算所有共同用户对的贡献
    for u in common_users_list:
        for v in common_users_list:
            if u == v:
                continue

            # 找到用户u和v的共同交互物品
            common_items_uv = user_items[u].intersection(user_items[v])

            # 使用预计算的用户权重
            user_weight_u = user_weights[u]  # 1.0 / sqrt(|I_u|)
            user_weight_v = user_weights[v]  # 1.0 / sqrt(|I_v|)

            # Swing Score核心公式
            contribution = (user_weight_u * user_weight_v) / (alpha + len(common_items_uv))
            swing_score += contribution

    return swing_score


def training_model():
    import funrec
    from funrec.utils import build_metrics_table

    # 加载配置
    config = funrec.load_config('swing')

    # 加载数据
    train_data, test_data = funrec.load_data(config.data)

    # 准备特征
    feature_columns, processed_data = funrec.prepare_features(config.features, train_data, test_data)

    # 训练模型
    models = funrec.train_model(config.training, feature_columns, processed_data)

    # 评估模型
    metrics = funrec.evaluate_model(models, processed_data, config.evaluation, feature_columns)

    print(build_metrics_table(metrics))


if __name__ == '__main__':
    training_model()
