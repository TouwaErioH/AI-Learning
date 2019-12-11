'''

test综合了生成model与调用model的代码

并包含测试部分，测试出最小误差的rank，iterations，lambda为 6,22,0.13

之后根据test编写了createmodel和callmodel


'''

# 使用Spark MLlib中推荐算法ALS对电影评分数据MovieLens推荐
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
import numpy as np #arange


def alsModelEvaluate(model, testing_rdd):
    # 对测试数据集预测评分，针对测试数据集进行预测
    predict_rdd = model.predictAll(testing_rdd.map(lambda r: (r[0], r[1])))
    #print(predict_rdd.take(5))
    predict_actual_rdd = predict_rdd.map(lambda r: ((r[0], r[1]), r[2])) \
        .join(testing_ratings.map(lambda r: ((r[0], r[1]), r[2])))

    #print(predict_actual_rdd.take(5))
    # 创建评估指标实例对象
    metrics = RegressionMetrics(predict_actual_rdd.map(lambda pr: pr[1]))

    print("MSE = %s" % metrics.meanSquaredError)
    print("RMSE = %s" % metrics.rootMeanSquaredError)

    # 返回均方根误差
    return metrics.rootMeanSquaredError


def train_model_evaluate(training_rdd, testing_rdd, rank, iterations, lambda_):
    # 定义函数，训练模型与模型评估
    # 使用超参数的值，训练数据和ALS算法训练模型
    print(lambda_,rank,iterations)
    model = ALS.train(training_rdd, rank, iterations, lambda_)

    # 模型的评估
    rmse_value = alsModelEvaluate(model, testing_rdd)

    # 返回多元组
    return (model, rmse_value, rank, iterations, lambda_)


if __name__ == "__main__":
    # 构建SparkSession实例对象
    # local 127.0.0.1
    spark = SparkSession.builder \
        .appName("SparkSessionExample") \
        .master("local") \
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext

    # 读取数据
    raw_ratings_rdd = sc.textFile("C:/Users/lenovo/Desktop/ml-100k/ml-100k/u.data")
    # print(raw_ratings_rdd.count())
    # print(raw_ratings_rdd.first())

    # 获取评分数据前三个字段，构建Rating实例对象
    ratings_rdd = raw_ratings_rdd.map(lambda line: line.split('\t')[0:3])
    # print(ratings_rdd.first())

    ratings_datas = ratings_rdd.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
    # print(ratings_datas.first())

    # 查看评分数据中有多少电影
    # print(ratings_datas.map(lambda x: x[1]).distinct().count())

    # 查看评分数据中有多少用户
    # print(ratings_datas.map(lambda x: x[0]).distinct().count())

    # 将数据集分为训练数据集和测试数据集
    training_ratings, testing_ratings = ratings_datas.randomSplit([0.7, 0.3])

    '''
        # 使用ALS算法来训练模型
        # help(ALS)
        # 采用显示评分函数训练模型
        alsModel = ALS.train(training_ratings, 10, iterations=10, lambda_=0.01)

        # 用户特征因子矩阵
        user_feature_matrix = alsModel.userFeatures()
        print(type(user_feature_matrix))
        print(user_feature_matrix.take(10))

        # 物品因子矩阵
        item_feature_matrix = alsModel.productFeatures()
        print(type(item_feature_matrix))
        print(item_feature_matrix.take(10))

        # 预测某个用户对某个电影的评分

        # 假设用户196，对电影242的评分，实际评分为3分

        predictRating = alsModel.predict(196, 242)
        print(predictRating)

        # 为用户推荐（10部电影）
        rmdMovies = alsModel.recommendProducts(196, 10)
        print(rmdMovies)

        # 为电影推荐（10个用户）
        rmdUsers = alsModel.recommendUsers(242, 10)
        print(rmdUsers)
    '''

    # 怎么评价模型的好坏，ALS模型评估指标(类似回归算法模型预测值，连续值)，使用回归模型中
    # RMSE（均方根误差）评估模型
    # 找到最佳模型
    '''
        如何找到最佳模型？？
            -a. 模型的评估
                计算RMSE
            -b. 模型的优化，两个方向
                1、数据
                2、超参数的调整，选择合适的超参数的值，得到最优模型
            交叉验证
                训练数据集、验证数据集、测试数据集
            K-Folds交叉验证
    '''

    # ALS算法的超参数的调整
    # 定义一个函数，用于对模型进行评估
    # 使用三层for循环，设置不同参数的值，分别使用ALS算法训练模型，评估获取RMSE的值
    metrix_list = [train_model_evaluate(training_ratings, testing_ratings,6,param_iterations, 0.13)
                   #for param_rank in range(5,10)
                   for param_iterations in [1,22]
                   #for param_rank in range(1, 10)
                   #for param_iterations in range(1, 10)
                   #for param_lambda in np.arange(0.09,0.14,0.01)
                   ]
    print(type(metrix_list))
    #sorted(metrix_list, key=lambda k: k[1], reverse=False)
    print("\n")
    print("\n")
    metrix_list.sort(key=lambda s: s[1])
    print(metrix_list)
    print("\n")
    model, rmse_value, rank, iterations, lambda_ = metrix_list[0]
    print("The best parameters, rank=%s, iterations=%s, lambda_=%s" % (rank ,iterations ,lambda_))

    # 保存模型
    #model.save(sc, "C:/Users/lenovo/Desktop/ml-100k/ml-100k/als_model")
    print(model)
    # 加载模型
    #load_model = MatrixFactorizationModel.load(sc, "C:/Users/lenovo/Desktop/ml-100k/ml-100k/als_model")