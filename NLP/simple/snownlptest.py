from snownlp import SnowNLP
from snownlp import sentiment

def train():
    sentiment.train('F:/Anaconda/Lib/site-packages/snownlp/sentiment/neg.txt','F:/Anaconda/Lib/site-packages/snownlp/sentiment/pos.txt')
    sentiment.save('F:/Anaconda/Lib/site-packages/snownlp/sentiment/sentiment2.marshal')

def test():
    wordlist=['没什么好看的',"内容很有趣","内容不错，但是插画不好看","内容不错，但是插画不好看，总的来说一般","内容不错但是插画不好看","令人失望"
        ,"辣鸡店主，败我钱财，毁我青春","快递很慢，差评","孩子喜欢，我不喜欢","孩子和我都喜欢","内容不错"]
    for word in wordlist:
        #word2=word.encode("unicode_escape")
        q = SnowNLP(word)
        print('%s:  %f\n'%(word,q.sentiments))
if __name__=='__main__':
        #train()
        test()


# q = SnowNLP(u'好重的味道，好难闻，而且严重掉色，请大家还是要看过再买吧')
# print('%d\n'%(q.sentiments))

