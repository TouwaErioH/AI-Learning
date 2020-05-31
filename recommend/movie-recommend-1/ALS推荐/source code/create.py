from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS

def SetPath(sc):
    """定义全局变量Path，配置文件读取"""
    global Path
    Path = "C:/Users/lenovo/Desktop/final"


def CreateSparkContext():
    """定义CreateSparkContext函数便于创建SparkContext实例"""
    sparkConf = SparkConf() \
             .setAppName("Recommend") \
             .set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    print("master="+sc.master)
    return sc

def PrepareData(sc):
    """数据预处理:读取u.data文件，转化为ratingsRDD数据类型"""
    rawUserData = sc.textFile(Path + "/u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    return ratingsRDD


def SaveModel(sc):
    """存储模型"""
    try:
        model.save(sc, Path+"/ALSmodel")
        print("模型已存储")
    except Exception:
        print("模型已存在,先删除后创建")
   


if __name__ == "__main__":
    sc = CreateSparkContext()
    print("==========数据准备阶段==========")
    ratingsRDD = PrepareData(sc)
    print("========== 训练阶段 ============")
    print(" 开始ALS训练，参数rank=6,iterations=22,lambda=0.13")
    model = ALS.train(ratingsRDD, 6, 22, 0.13)
    print("========== 存储model ==========")
    SaveModel(sc)
    sc.stop()  #退出已有SparkContext