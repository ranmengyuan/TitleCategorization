# coding=utf-8
import jieba.posseg as pseg
from dataBase.toDatabase import Headline, read_by_line, data_todatabase, create_data, get_element, create_result, \
    result_todatabase, conn_db
from Analyze.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from tgrocery import Grocery
import xgboost as xgb
# from sklearn.cross_validation import train_test_split

import numpy as np

wordtype = []
wordtype.append('n')
wordtype.append('i')
wordtype.append('l')
wordtype.append('Ng')
wordtype.append('nr')
wordtype.append('ns')
wordtype.append('nt')
wordtype.append('nz')
wordtype.append('v')
wordtype.append('vg')
wordtype.append('vd')
wordtype.append('vn')
wordtype.append('a')
wordtype.append('ag')
wordtype.append('ad')
wordtype.append('an')

target = []
target.append("history")
target.append("military")
target.append("baby")
target.append("world")
target.append("tech")
target.append("game")
target.append("society")
target.append("sports")
target.append("travel")
target.append("car")
target.append("food")
target.append("entertainment")
target.append("finance")
target.append("fashion")
target.append("discovery")
target.append("story")
target.append("regimen")
target.append("essay")


def get_stop():
    """
    获取停用词典
    :return:
    """
    stopwords = {}
    fstop = open('//Volumes//Transcend//文件//实验室//标题分类//chinese_stopword.txt', 'r')
    for eachWord in fstop:
        stopwords[eachWord.strip()] = eachWord.strip()
    fstop.close()
    return stopwords


#

def chang_result(word):
    """
    获得结果的序号
    :return:
    """
    for i in range(len(target)):
        if word == target[i]:
            return i
    return -1


# def get_data():
#     """
#     获取数据,整理后存入数据库
#     :return:
#     """
#     stop = get_stop()
#     stopwords = {}.fromkeys(stop)
#     x_train = []
#     y_train = []
#     x_test = []
#     file_content = read_by_line("//Volumes//Transcend//文件//实验室//标题分类//nlpcc_data//word//train.txt")
#     for i in range(len(file_content)):
#         content = file_content[i].split("\n")
#         temp_data = content[0].split("\t")
#         temp = temp_data[1].split(" ")
#         j = 0
#         x = []
#         y = []
#         index = chang_result(temp_data[0])
#         y_train.append(index)
#         while 1:
#             if j >= len(temp):
#                 break
#             if temp[j] not in stopwords:
#                 x.append(temp[j])
#             j += 1
#         x_train.append(x)
#
#     file_content = read_by_line("//Volumes//Transcend//文件//实验室//标题分类//test//test.word")
#     for i in range(len(file_content)):
#         content = file_content[i].split("\n")
#         temp = content[0].split(" ")
#         j = 0
#         x = []
#         while 1:
#             if j >= len(temp):
#                 break
#             if temp[j] not in stopwords:
#                 x.append(temp[j])
#             j += 1
#         x_test.append(x)
#     return x_train, y_train, x_test


def get_data():
    """
    获取数据,整理后存入数据库
    :return:
    """
    stop = get_stop()
    stopwords = {}.fromkeys(stop)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    test = []
    file_content = read_by_line("//Volumes//Transcend//文件//实验室//标题分类//nlpcc_data//word//train.txt")
    # conn, cur = create_data("Train")
    for i in range(len(file_content)):
        content = file_content[i].split("\n")
        temp_data = content[0].split("\t")
        temp = temp_data[1].split(" ")
        j = 0
        x = []
        y = []
        index = chang_result(temp_data[0])
        y_train.append(index)
        while 1:
            if j >= len(temp):
                break
            if temp[j] not in stopwords:
                # data = Headline()
                # data.result = temp_data[0]
                # data.content = temp[j]
                # data.sentence_id = i + 1
                # data_todatabase(conn, cur, data, "Train")
                x.append(temp[j])
            j += 1
        x_train.append(x)
        # y_train.append(y)

    # conn, cur = create_data("Test")
    file_content = read_by_line("//Volumes//Transcend//文件//实验室//标题分类//nlpcc_data//word//dev.txt")
    for i in range(len(file_content)):
        content = file_content[i].split("\n")
        temp_data = content[0].split("\t")
        temp = temp_data[1].split(" ")
        j = 0
        x = []
        y = []
        index = chang_result(temp_data[0])
        y_test.append(index)
        t = ''
        while 1:
            if j >= len(temp):
                break
            t += temp[j]
            if temp[j] not in stopwords:
                # data = Headline()
                # data.result = temp_data[0]
                # data.content = temp[j]
                # data.sentence_id = i + 1
                # data_todatabase(conn, cur, data, "Test")
                x.append(temp[j])
            j += 1
        x_test.append(x)
        test.append(t)
        # y_test.append(y)
    return x_train, y_train, x_test, y_test, test


# def get_data():
#     """
#     获取数据,并整理
#     :return:
#     """
#     stopwords = get_stop()
#     x_train = []
#     y_train = []
#     x_test = []
#     y_test = []
#     test = []
#     file_content = read_by_line("//Volumes//Transcend//文件//实验室//标题分类//nlpcc_data//sentence//train.txt")
#     for i in range(len(file_content)):
#         content = file_content[i].split("\n")
#         temp_data = content[0].split("\t")
#         temp = pseg.cut(temp_data[1])
#         x = []
#         y_train.append(temp_data[0])
#         for w in temp:
#             if (w.word not in stopwords) & (w.flag in wordtype):
#                 x.append(w.word)
#         x_train.append(x)
#
#     file_content = read_by_line("//Volumes//Transcend//文件//实验室//标题分类//nlpcc_data//sentence//test.txt")
#     for i in range(len(file_content)):
#         content = file_content[i].split("\n")
#         temp_data = content[0].split("\t")
#         temp = pseg.cut(temp_data[1])
#         x = []
#         y_test.append(temp_data[0])
#         for w in temp:
#             if (w.word not in stopwords) & (w.flag in wordtype):
#                 x.append(w.word)
#         x_test.append(x)
#         test.append(temp_data[1])
#     return x_train, y_train, x_test, y_test, test


def analyze_data():
    """
    分析数据,获得每个元素的概率
    :return:
    """
    # conn, cur = create_result("Rate")
    conn = conn_db()
    cur = conn.cursor()
    i = 89
    while 1:
        element = get_element(cur, i)
        if len(element) == 0:
            break
        for j in range(len(element)):
            print(element[j])
            sum, count = MultinomialNB(element[j], target)
            if sum == 0:
                for n in range(18):
                    rate.append(0)
            else:
                rate = []
                for k in range(len(count)):
                    temp = count[k] / sum
                    rate.append(temp)
            result_todatabase(conn, cur, element[j], i, rate, "Rate")

        i += 1


if __name__ == "__main__":
    target = []
    target.append("history")
    target.append("military")
    target.append("baby")
    target.append("world")
    target.append("tech")
    target.append("game")
    target.append("society")
    target.append("sports")
    target.append("travel")
    target.append("car")
    target.append("food")
    target.append("entertainment")
    target.append("finance")
    target.append("fashion")
    target.append("discovery")
    target.append("story")
    target.append("regimen")
    target.append("essay")

    # get_data()
    # analyze_data()

    x_train, y_train, x_test, y_test, test = get_data()
    print(len(x_train))
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.9, random_state=4242)
    # print(len(x_train))

    count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english', tokenizer=lambda doc: doc,
                                lowercase=False)
    x_train = count_vec.fit_transform(x_train)
    train = x_train.toarray()
    x_test = count_vec.transform(x_test)
    test = x_test.toarray()

    # clf = MultinomialNB(alpha=0.1).fit(x_train, y_train)
    # doc_class_predicted = clf.predict(x_test)
    # f = open('resultdata.text', 'a')
    # for i in range(len(doc_class_predicted)):
    #     index = int(doc_class_predicted[i])
    #     f.write(target[index] + "\n")

    # dtrain = xgb.DMatrix(train, label=y_train)
    # dtest = xgb.DMatrix(test)
    # # param = {'max_depth': 6, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1, 'objective': 'multi:softmax',
    # #          'num_class': 3}  # 参数
    # # param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'nthread': 4, 'min_child_weight': 5,
    # #     'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'eta': 0.05,
    # #     'silent': 1, 'objective': 'binary:logistic'}
    # param = {'learning_rate': 0.1, 'max_depth': 3, 'eta': 0.05, 'eval_metric': 'merror',
    #          'silent': 1, 'objective': 'multi:softmax', 'num_class': 18}  # 参数
    #
    # evallist = [(dtrain, 'train')]  # 这步可以不要，用于测试效果
    # num_round = 50  # 循环次数
    # plst = param.items()
    # bst = xgb.train(plst, dtrain, num_round, evallist)
    # preds = bst.predict(dtest)
    # print(np.mean(preds == y_test))
    # for value in preds:
    #     print(value)


    grocery = tgrocery.Grocery('sample')
    grocery.train(train, label=y_train)
    grocery.save()
    grocery.load()
    preds = grocery.predict(x_test)
    print(preds)



# i = 0
# f = open('result3.text', 'a')
# result = []
# for pre in doc_class_predicted:
#     result.append(pre)
#     f.write(test[i] + "\t" + pre + "\n")
#     i += 1
# for i in range(len(target)):
#     n = result.count(target[i])
#     n1 = y_test.count(target[i])
#     print(target[i] + "\t" + str(n) + "\t" + str(n1))


# precision, recall, thresholds = precision_recall_curve(y_test, doc_class_predicted)
# answer = clf.predict_proba(x_test)[:, 1]
# report = answer > 0.5
# print(classification_report(y_test, report, target_names=['neg', 'pos']))

# x = np.load("a.npy")
# y = np.load("b.npy")
# y_train = y.ravel()
# y = np.array(y_train).astype(int)
# clf = MultinomialNB(alpha=0.1)
# clf.fit(x, y)
# pred = clf.predict([x[0]])
# print(pred)
