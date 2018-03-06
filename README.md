# TitleCategorization

TitleCategorization是一款对新闻标题进行分类的工具。给定新闻标题x =（x1，x2，...，xn），其中xj表示x中的第j个字，对象是找出其可能的类别或标签c∈C。更具体地说，我们需要找出一个函数来预测x属于哪个类别。主要是通过TF-IDF对标题的关键词进行提取，然后通过朴素贝叶斯、SVM、Xgboost对文本进行分类。

# 入门

TitleCategorization包括analyze、bean、dataBase、file和main。

file主要是文件操作，对给定的已知数据进行读取，建立训练集。

bean和dataBase主要是对数据库进行操作。

analyze主要是通过算法对数据进行预测。

main主要是对整个程序进行控制。

# 文件结构

bean和dataBase主要是对数据库进行操作。

analyze主要是通过算法对数据进行预测。

control.py主要是对整个程序进行控制。

result.text是正确的结果。

result1.text是用朴素贝叶斯分类算法后得到的结果。

result2.text是用SVM算法后得到的结果。

result3.text是用Xgboost算法后得到的结果。

static.text是用朴素贝叶斯分类算法后得到的结果与正确结果各类标题数量的对比。

static1.text是用Svm算法后得到的结果与正确结果各类标题数量的对比。

static2.text是用Xgboost算法后得到的结果与正确结果各类标题数量的对比。

resultdata是三个分类算法进行投票机制融合后的最终预测结果。

# 支持平台

TitleCategorization基于Python3.5。如果想要运行TitleCategorization推荐下载Python3.x解析器，并且需要pymysql，sklearn等包的支持。同时，需要注意处理文件和网页的格式。

# 疑问

如果您发现了诸如崩溃、意外行为或类似的问题，请访问[issue tracker](https://github.com/ranmengyuan/TitleCategorization/issues)方便交流。

谢谢！
