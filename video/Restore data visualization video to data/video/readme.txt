文件结构：

video.mp4: 源视频

get_frame_basic: 按固定间隔截图片
   |- getframe_by_seconds.py: 截取的代码
   |- 1965final Q4.jpg-2019final Q4.jpg: 截取的结果
   |- video.mp4: 要截取的视频

get_frame_final: 最后使用的自动判断的截取图片
   |- judged: 里面保存截取图片的结果
   |- autojudge.py: 自动判断截取图片的代码
   |- video.mp4: 要截取的视频

read_text_baiduapi: 使用百度OCR API高精度模式识别
   |- 1965.jpg-2019.jpg: 待识别的图片
   |- read_text.py: 进行识别的代码
   |- testf.csv: 由识别的结果的txt文件转化成的csv文件

read_text_tesseract: 使用pytesseract进行识别
   |- result: 不同版本的tesseract识别图片的结果（仅包含对切割后的图片的识别结果）
      |- tesseract5.0fail: tesseract5.0使用whitelist识别的结果
      |- result.csv: 对数据进行清洗之后得到的数据
      |- result3.0fail.txt: tesseract3.0使用whitelist识别的结果
      |- result4.0.txt: tesseract4.0（不支持whitelist）识别的结果
   |- judged_binary.rar: 将黑白图片变成切割后的二值图片，该版本使用judged_clean中的图片作为输入
   |- judged_binary_final.rar: 将黑白图片变成切割后的二值图片，该版本为优化后的最终版本，使用judged_clean_final中的图片作为输入
   |- judged_clean.rar: 将彩色图片处理成黑白图片，该版本有部分字符处理后不清晰
   |- judged_clean_final.rar: 将彩色图片处理成黑白图片，该版本修改了过滤条件对judged_clean进行了优化
   |- judged_fail1_block.rar: 将彩色图片处理成黑白图片，该版本矩形框没有去除
   |- judged_fail2_100.rar: 将彩色图片处理成黑白图片，该版本使用R通道小于100作为判断条件
   |- judged_fail3_90.rar: 将彩色图片处理成黑白图片，该版本使用R通道小于90作为判断条件
   |- judged_failcrop_10.rar: 将黑白图片二值化并进行切割，但是认为差距为10像素的时候是有较大空白，会出现例如visual basic这种有空格的词被切成两个

view: 动态展示
   |- 1th popular language.png-5th popular language.png: 用testf.csv画出55年来第k流行的折线图
   |- Figure_1.png-Figure_55.png: 用testf.csv画出的每年的直方图
   |- histogram.avi: 由生成的直方图组成的视频
   |- image_to_video.py: 将指定图片转化为视频的代码
   |- line chart.avi: 由折线图生成的视频
   |- project.avi: 由截取的图片组成的视频
   |- testf.csv: 识别获得的数据
   |- view.py: 画直方图，折线图等的代码


other: 其他文件，可供参考；autogetbasic的基础上完成了getframefinal
optimize.txt: 可改进之处