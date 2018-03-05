from dataBase.sql_helper import conn_db, conn_close, exe_query, exe_table, exe_update


class Headline:
    content = ''
    sentence_id = 0
    result = ''

    def __init__(self):
        self.content = ''
        self.sentence_id = 0
        self.result = ''


def read_by_line(address):
    """
    带缓存的文件读取一行数据
    :param address:
    :return:
    """
    file = open(address)
    file_content = []
    while 1:
        lines = file.readlines(10000)
        if not lines:
            break
        for line in lines:
            file_content.append(line)
    return file_content


def create_data(form):
    """
    建立数据库表格
    :param form:
    :return:
    """
    try:
        conn = conn_db()
        cur = conn.cursor()
        sql = "DROP TABLE if EXISTS " + form
        exe_table(cur, sql)
        sql = "CREATE TABLE " + form + "(id INT NOT NULL AUTO_INCREMENT,content VARCHAR (255),sentence_id INT ," \
                                       "result VARCHAR(255) ,PRIMARY KEY (id)) ENGINE = InnoDB DEFAULT CHARSET = UTF8"
        exe_table(cur, sql)

    except Exception as e:
        print(e)
    finally:
        return conn, cur


def data_todatabase(conn, cur, data, form):
    """
    将文本文件处理并存入数据库中
    :param file_content:
    :return:
    """
    try:
        sql = "INSERT INTO " + form + "(content,sentence_id,result) VALUES ('" + data.content + "','" + str(
                data.sentence_id) + "','" + data.result + "')"
        print(form + "\t" + data.content)
        exe_table(cur, sql)
    except Exception as e:
        print(e)


def create_result(form):
    """
    建立数据库表格
    :param form:
    :return:
    """
    try:
        conn = conn_db()
        cur = conn.cursor()
        sql = "DROP TABLE if EXISTS " + form
        exe_table(cur, sql)
        sql = "CREATE TABLE " + form + "(id INT NOT NULL AUTO_INCREMENT,content VARCHAR (255),sentence_id INT ," \
                                       "history float ,military float ,baby float ,world float ,tech float ,game float " \
                                       ",society float ,sports float ,travel float ,car float ,food float ,entertainment float " \
                                       ",finance float ,fashion float ,discovery float ,story float ,regimen float ,essay float " \
                                       ",PRIMARY KEY (id)) ENGINE = InnoDB DEFAULT CHARSET = UTF8"
        exe_table(cur, sql)

    except Exception as e:
        print(e)
    finally:
        return conn, cur


def result_todatabase(conn, cur, content, sentence, rate, form):
    """
    将文本文件处理并存入数据库中
    :param file_content:
    :return:
    """
    try:
        sql = "INSERT INTO " + form + "(content,sentence_id,history ,military ,baby ,world ,tech ,game " \
                                      ",society ,sports ,travel ,car ,food ,entertainment " \
                                      ",finance ,fashion ,discovery ,story ,regimen ,essay) VALUES ('" + content + \
              "','" + str(sentence) + "','" + str(rate[0]) + "','" + str(rate[1]) + "','" + str(rate[2]) + "','" + str(
                rate[3]) + "','" \
              + str(rate[4]) + "','" + str(rate[5]) + "','" + str(rate[6]) + "','" + str(rate[7]) + "','" + str(
                rate[8]) + "','" + str(rate[9]) + \
              "','" + str(rate[10]) + "','" + str(rate[11]) + "','" + str(rate[12]) + "','" + str(
                rate[13]) + "','" + str(rate[14]) + "'," \
                                                    "'" + str(rate[
                                                                  15]) + "','" + str(rate[16]) + "','" + str(
                rate[17]) + "')"
        exe_update(conn, cur, sql)

    except Exception as e:
        print(e)


def get_sum(cur, data):
    """
    获得某个元素出现的总次数
    :param cur:
    :param data:
    :return:
    """
    temp = -1
    try:
        sql = "SELECT COUNT(content)FROM Train WHERE content='" + data + "'"
        sum = exe_query(cur, sql)
        temp = ""
        for num in sum:
            temp = int(num[0])
    except Exception as e:
        print(e)
        temp = -1
    finally:
        return temp


def get_count(cur, data, target):
    """
    获得某个元素在不同条件下的出现次数
    :param cur:
    :param data:
    :param target:
    :return:
    """
    count = []
    try:
        for i in range(len(target)):
            sql = "SELECT COUNT(content)FROM Train WHERE content='" + data + "' AND result='" + target[i] + "'"
            num = ""
            temp = exe_query(cur, sql)
            for en in temp:
                num = int(en[0])
            count.append(num)
    except Exception as e:
        print(e)
        count = []
    finally:
        return count


def get_element(cur, sentence):
    """
    获得一个句子的分词结果
    :param cur:
    :param sentence:
    :return:
    """
    element = []
    try:
        sql = "SELECT content FROM Test WHERE sentence_id=" + str(sentence)
        temp = exe_query(cur, sql)
        for en in temp:
            element.append(str(en[0]))
    except Exception as e:
        print(e)
        element = []
    return element
