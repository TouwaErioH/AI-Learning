from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys


def SetPath(sc):
    """定义全局变量Path，配置文件读取"""
    global Path
    Path = "C:/Users/lenovo/Desktop/ml-100k/ml-100k"


def CreateSparkContext():
    """定义CreateSparkContext函数便于创建SparkContext实例"""
    sparkConf = SparkConf() \
             .setAppName("Recommend") \
             .set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    print("master="+sc.master)
    return sc


def loadModel(sc):
    """载入训练好的推荐模型"""
    try:
        model = MatrixFactorizationModel.load(sc, Path+"ALSmodel")
        print("载入模型成功")
    except Exception:
        print("模型不存在, 请先训练模型")
    return model

def PrepareData(sc):
    """数据准备：准备u.item文件返回电影id-电影名字典（对照表）"""
     #movieTitle为dict类型
    itemRDD = sc.textFile(Path+"/u.item")
    movieTitle = itemRDD.map(lambda line: line.split("|")) \
        .map(lambda a: (int(a[0]), a[1])) \
        .collectAsMap()
    return movieTitle


#没有做去重工作，即已经打过分的电影不推荐 （不清楚recommendProducts是否自动去重，可以阅读源码/官方文档分析一下）
def RecommendUsers(model,movieTitle,inputname,inputcount):
    RecommendUser = model.recommendProducts(inputname,inputcount)
    print("对ID为"+str(inputname)+"的用户推荐下列"+str(inputcount)+"个电影：")
    for p in RecommendUser:
        mark=str(p[2])
        mark=float(p[2])
        if mark>5:
            mark=5.000123
        print("对编号为" + str(p[0]) + "的用户" + "推荐电影" + str(movieTitle[p[1]]) + "\n")
        print("推荐评分为 %.1f \n"  % mark)
    #sc.stop()  #退出已有SparkContext

def RecommendMovies(model,movieTitle,inputname,inputcount):
    RecommendUser = model.recommendUsers(inputname,inputcount)
    print("将ID为"+str(inputname)+"的电影推荐给下列"+str(inputcount)+"用户：")
    for p in RecommendUser:
        mark=str(p[2])
        mark=float(p[2])
        if mark>5:
            mark=5.000123
        print("对编号为" + str(p[0]) + "的用户" + "推荐电影" + str(movieTitle[p[1]]) + "\n")
        print("推荐评分为 %.1f \n"  % mark)
    #sc.stop()  #退出已有SparkContext


if __name__ == "__main__":
    '''
    print("请输入2个参数, 第一个参数指定推荐模式（用户/电影）, 第二个参数为推荐的数量如U666 10表示向用户666推荐10部电影")
    input = ["U666",'5']
    '''
    print("请输入2个参数, 第一个参数指定推荐用户, 第二个参数为推荐的数量.如666 10表示向用户666推荐10部电影")
    sc.stop()
    #input = ["666",'5']
    username='666'
    usercount='10'
    userreal=int(username)
    moviecount=int(usercount)
    sc=CreateSparkContext()
    print("==========数据准备==========")
    movieTitle = PrepareData(sc)
    print("==========载入模型==========")
    model = loadModel(sc)
    print("==========进行推荐==========")
    #Recommend(model)
    RecommendUsers(model, movieTitle, userreal,moviecount)
    
    
    userreal=123
    moviecount=5
    RecommendMovies(model, movieTitle, userreal,moviecount)
    sc.stop()
