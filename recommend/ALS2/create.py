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
    predict_actual_rdd = predict_rdd.map(lambda r: ((r[0], r[1]), r[2])) \
        .join(testing_ratings.map(lambda r: ((r[0], r[1]), r[2])))

    # 创建评估指标实例对象
    metrics = RegressionMetrics(predict_actual_rdd.map(lambda pr: pr[1]))

    #print("MSE = %s" % metrics.meanSquaredError)
    #print("RMSE = %s" % metrics.rootMeanSquaredError)

    # 返回均方根误差
    return metrics.rootMeanSquaredError


def train_model_evaluate(training_rdd, testing_rdd, rank, iterations, lambda_):
    # 定义函数，训练模型与模型评估
    # 使用超参数的值，训练数据和ALS算法训练模型
    #print(lambda_,rank,iterations)
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

    # 将数据集分为训练数据集和测试数据集
    training_ratings, testing_ratings = ratings_datas.randomSplit([0.7, 0.3])

    #生成模型
    metrix_list = [train_model_evaluate(training_ratings, testing_ratings,6,param_iterations, 0.13)
                   for param_iterations in [1,22]     
                   ]
    print(type(metrix_list))    #已经确定了rank，iterations，lambda为6,22,0.13。加一个1是因为直接指定经常报错。原因可能是栈溢出（越界访问）
    #sorted(metrix_list, key=lambda k: k[1], reverse=False)
    print("\n")
    print("\n")
    metrix_list.sort(key=lambda s: s[1])
    #print(metrix_list)
    print("\n")
    model, rmse_value, rank, iterations, lambda_ = metrix_list[0]
    #print("The best parameters, rank=%s, iterations=%s, lambda_=%s" % (rank ,iterations ,lambda_))

    # 保存模型
    model.save(sc, "C:/Users/lenovo/Desktop/ml-100k/ml-100k/als_model")
    print(model)
