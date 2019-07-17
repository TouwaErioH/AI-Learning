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
    '''
    # 读取数据
    raw_ratings_rdd = sc.textFile("C:/Users/lenovo/Desktop/ml-100k/ml-100k/u.data")
    # print(raw_ratings_rdd.count())
    # print(raw_ratings_rdd.first())

    # 获取评分数据前三个字段，构建Rating实例对象
    ratings_rdd = raw_ratings_rdd.map(lambda line: line.split('\t')[0:3])
    # print(ratings_rdd.first())

    ratings_datas = ratings_rdd.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
    # print(ratings_datas.first())


    # 将数据集分为训练数据集和测试数据集
    training_ratings, testing_ratings = ratings_datas.randomSplit([0.7, 0.3])
    '''
    
    
    # 加载模型
    alsModel = MatrixFactorizationModel.load(sc, "C:/Users/lenovo/Desktop/ml-100k/ml-100k/als_model")

        # 用户特征因子矩阵
    user_feature_matrix = alsModel.userFeatures()
    print(type(user_feature_matrix))
    print("\n")
    print(user_feature_matrix.take(10))
    print("\n")
        # 物品因子矩阵
    item_feature_matrix = alsModel.productFeatures()
    print(type(item_feature_matrix))
    print("\n")
    print(item_feature_matrix.take(10))

        # 预测某个用户对某个电影的评分

        # 假设用户196，对电影242的评分，实际评分为3分
    print("\n")
    predictRating = alsModel.predict(196, 242)
    print(predictRating)
    
    print("\n")

        # 为用户推荐（10部电影）
    rmdMovies = alsModel.recommendProducts(196, 10)
    print(rmdMovies)
    
    print("\n")
        # 为电影推荐（10个用户）
    rmdUsers = alsModel.recommendUsers(242, 10)
    print(rmdUsers)


