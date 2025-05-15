#!/usr/bin/env python3
import os
import random

# 设置随机种子（可选），以保证结果可复现
random.seed(42)


def read_file(filename):
    """
    读取文件，返回字典： { user_id: set(物品ID) }
    文件格式：每一行，第一个数字为用户ID，后续为该用户交互的物品ID（以空格分隔）
    """
    user_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            user = tokens[0]
            items = tokens[1:]
            if user not in user_dict:
                user_dict[user] = set()
            # 加入当前行的所有物品ID
            user_dict[user].update(items)
    return user_dict


def write_file(filename, data):
    """
    将字典写入文件，格式与原文件相同：
    每一行：用户ID 物品ID1 物品ID2 ...
    """
    with open(filename, 'w') as f:
        for user in sorted(data, key=lambda x: int(x)):
            # 如果某个用户对应的列表为空，则只写用户ID
            if data[user]:
                line = user + " " + " ".join(data[user]) + "\n"
            else:
                line = user + "\n"
            f.write(line)


def split_user_interactions(items, ratio=(0.6, 0.2, 0.2)):
    """
    对单个用户的交互列表 items 进行随机打乱后按照比例划分成训练/验证/测试三部分。
    当交互总数 total>=3时，确保每部分至少 1 个；否则返回全部归入训练集。
    返回：train_list, valid_list, test_list（列表中的元素均为字符串）
    """
    total = len(items)
    if total < 3:
        # 交互较少时，不进行划分
        return items, [], []

    random.shuffle(items)
    # 定义划分点，确保至少每部分有 1 个
    train_end = max(1, int(total * ratio[0]))
    valid_end = max(train_end + 1, int(total * (ratio[0] + ratio[1])))
    if valid_end >= total:
        valid_end = total - 1

    train_items = items[:train_end]
    valid_items = items[train_end:valid_end]
    test_items = items[valid_end:]
    return train_items, valid_items, test_items


def main():
    # 文件名（请确保脚本所在目录有 train.txt 与 test.txt）
    train_filename = "train.txt"
    test_filename = "test.txt"

    # 读取原始文件，得到每个用户的交互集合
    train_data = read_file(train_filename)
    test_data = read_file(test_filename)

    # 合并两个字典：对于每个用户取并集（注意：这里以字符串形式保存物品ID）
    all_users = set(train_data.keys()) | set(test_data.keys())
    combined_data = {}
    for user in all_users:
        items = set()
        if user in train_data:
            items.update(train_data[user])
        if user in test_data:
            items.update(test_data[user])
        # 转换为列表便于后续随机划分
        combined_data[user] = list(items)

    # 初始化新的训练/验证/测试数据字典
    new_train, new_valid, new_test = {}, {}, {}
    for user, items in combined_data.items():
        tr, va, te = split_user_interactions(items, ratio=(0.6, 0.2, 0.2))
        # 为保证写入时顺序一致，将列表中的物品ID按数值排序后再转为字符串
        new_train[user] = sorted(tr, key=lambda x: int(x))
        new_valid[user] = sorted(va, key=lambda x: int(x))
        new_test[user] = sorted(te, key=lambda x: int(x))

    # 写入新的文件：这里将覆盖原来的 train.txt 与 test.txt，同时生成 valid.txt
    write_file("train.txt", new_train)
    write_file("valid.txt", new_valid)
    write_file("test.txt", new_test)
    print("数据已成功划分为 6:2:2，并写入 train.txt, valid.txt, test.txt 文件中。")


if __name__ == "__main__":
    main()
