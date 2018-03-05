from dataBase.toDatabase import conn_db, get_count, get_sum


def MultinomialNB(data, target):
    """
    每个样本确定每个标签集的正确预测。
    :param data:
    :param target:
    :return:
    """
    conn = conn_db()
    cur = conn.cursor()
    sum = get_sum(cur, data)
    count = get_count(cur, data, target)
    return sum, count
