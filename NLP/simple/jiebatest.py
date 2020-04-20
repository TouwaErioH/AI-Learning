import jieba

jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持

words='每当我遇到自己不敢直视的困难时，我就会闭上双眼，想象自己是一个80岁的老人，为人生中曾放弃和逃避过的无数困难而懊悔不已，我会对自己说，能再年轻一次该有多好，然后我睁开眼睛：砰！我又年轻一次了！'

#全模式
wordlist=jieba.cut(words, cut_all=True)
print("|".join(wordlist))

#精确模式
wordlist=jieba.cut(words)#cut_all=Flase
print("|".join(wordlist))

#搜索引擎模式
wordlist=jieba.cut_for_search(words)
print('|'.join(wordlist))

#paddle模式
wordlist=jieba.cut(words,use_paddle=True) # 使用paddle模式
print('|'.join(wordlist))
