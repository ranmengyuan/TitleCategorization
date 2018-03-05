import pymysql

def conn_db():
    """
    连接数据库
    """
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='B221gt12345', db='news', charset='UTF8')
    return conn


def exe_table(cur, sql):
    """
    创建表格或删除
    :param cur:
    :param sql:
    :return:
    """
    sta = cur.execute(sql)
    return sta


def exe_update(conn, cur, sql):
    """
    更新或插入操作或删除操作
    :param conn
    :param cur
    :param sql
    """
    sta = cur.execute(sql)
    # "delete from exe where Id=%d" % (int(eachID))
    conn.commit()
    return sta


def exe_query(cur, sql):
    """
    查找操作
    :param cur
    :param sql
    """
    cur.execute(sql)
    return cur


def conn_close(conn, cur):
    """
    关闭连接，释放资源
    :param conn
    :param cur
    """
    cur.close()
    conn.close()
