import numpy as np


def get_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def get_tooth_midline(boxes):
    """确定中轴参考线 X (同前，作为起始搜索基准)"""
    if len(boxes) == 0: return 0
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    for ref_label in range(1, 9):
        indices = np.where(boxes[:, 4] == ref_label)[0]
        if len(indices) >= 2:
            return np.mean(sorted(centers_x[indices]))
        if len(indices) == 1:
            ref_x = centers_x[indices[0]]
            others_x = centers_x[np.where(boxes[:, 4] != ref_label)[0]]
            if len(others_x) == 0: return ref_x
            closest_x = others_x[np.abs(others_x - ref_x).argmin()]
            return ref_x + (closest_x - ref_x) * 0.1 if ref_x < closest_x else ref_x - (ref_x - closest_x) * 0.1
    return np.mean(centers_x)


def build_robust_chain(pool, midline_x):
    """
    改进的链式追踪：防止跳牙
    """
    if len(pool) == 0: return np.array([]).reshape(0, 5)

    remaining = pool.tolist()
    chain = []

    # 1. 初始点：离中线 X 距离最近的作为“种子”
    centers_x = np.array([(b[0] + b[2]) / 2 for b in remaining])
    curr_idx = np.abs(centers_x - midline_x).argmin()
    current = remaining.pop(curr_idx)
    chain.append(current)

    # 2. 迭代寻找下一颗
    while len(remaining) > 0:
        curr_center = get_center(current)

        # 计算所有剩余牙齿到当前牙齿的：
        # a) 欧氏距离
        # b) 到中线的 X 轴距离 (用于判断谁更靠内)
        dists = []
        for i, cand in enumerate(remaining):
            cand_center = get_center(cand)
            d_phys = np.linalg.norm(curr_center - cand_center)
            # 衡量这颗牙离中线的距离（绝对值）
            d_to_midline = abs((cand[0] + cand[2]) / 2 - midline_x)
            dists.append((i, d_phys, d_to_midline))

        # 核心逻辑：排序策略
        # 我们不只按距离排，我们按 (物理距离 + 权重 * 离中线的远近) 排
        # 这样即使 4 比 3 稍微近一点，但因为 3 离中线更近，3 会被优先排在前面
        # 权重 0.5-1.0 是一个经验值，保证了“步进性”
        dists.sort(key=lambda x: x[1] + 0.8 * x[2])

        # 取排序后的第一个作为下一颗
        best_match_idx = dists[0][0]
        current = remaining.pop(best_match_idx)
        chain.append(current)

    return np.array(chain)


def rectify_label_logic(chain):
    """编号矫正：保持 1-8 递增"""
    if len(chain) == 0: return chain
    res = chain.copy()
    last_label = 0
    for i in range(len(res)):
        curr_label = int(res[i, 4])
        if curr_label <= last_label:
            curr_label = last_label + 1
        if curr_label > 8: curr_label = 8
        res[i, 4] = curr_label
        last_label = curr_label
    return res


def toothNumberingCorrection(boxes):
    """
    完整封装
    """
    if len(boxes) == 0: return boxes

    midline_x = get_tooth_midline(boxes)

    # 分左右
    c_x = (boxes[:, 0] + boxes[:, 2]) / 2
    left_pool = boxes[c_x < midline_x]
    right_pool = boxes[c_x >= midline_x]

    # 构建鲁棒链条
    left_chain = build_robust_chain(left_pool, midline_x)
    right_chain = build_robust_chain(right_pool, midline_x)

    # 矫正
    left_final = rectify_label_logic(left_chain)
    right_final = rectify_label_logic(right_chain)

    left_final[:, -1]=left_final[:, -1] +8
    right_final[:, -1] = 9-right_final[:, -1]

    return np.concatenate([left_final, right_final], axis=0)