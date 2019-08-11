import datetime
import re
import math
import collections
import numpy as np
import jieba
import wordcloud
from PIL import Image
import matplotlib.pyplot as plt

##聊天时间段统计
def len_of_chat(text):
    with open(text, 'rb') as f:
        firstTimeLine = str(f.readline())

        offset = 0
        while True:
            f.seek(offset,2)##从文本最后往前读，每次偏移-offset个字节
            a = f.read(1)
            if a == b'\n':
                lines = f.readlines()
                if len(lines) > 2:
                    lastTimeLine = str(lines[-2])
                    break
            offset = offset - 1

    time_start = re.search("20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", firstTimeLine).group()
    time_end = re.search("20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", lastTimeLine).group()

    start = time_start.split()
    date1 = start[0].split("-")
    time1 = start[1].split(":")

    end = time_end.split()
    date2 = end[0].split("-")
    time2 = end[1].split(":")

    t1 = datetime.datetime(int(date1[0]),int(date1[1]),int(date1[2]),int(time1[0]),int(time1[1]),int(time1[2]))
    t2 = datetime.datetime(int(date2[0]),int(date2[1]),int(date2[2]),int(time2[0]),int(time2[1]),int(time2[2]))

    return (t2-t1).days, (t2-t1).seconds


##按照用户名称，统计每个人的聊天总条数，总字数，单条微信最大字数和索引处
def count_of_chat(text, user):
    re_exp = user + " 20[0-9]{2}-[0-9]{2}-[0-9]{2}"##正则匹配 微信名字+时间戳，表示一条微信
    flag = 0
    count = 0
    words = 0
    max_words = 0
    with open(text, "r", encoding="utf-8") as f:
        for index,line in enumerate(f):
            if re.match(re_exp, line):
                count = count + 1
                flag = 1
            else:
                if flag == 1 and line.split() != []:
                    flag = 0
                    new_len = len(line)
                    words = words + new_len
                    if new_len > max_words:
                        max_words = new_len
                        max_index = index
                
    return count, words, max_words, max_index


##计算在统计时间段中，有多少天在聊天
def day_of_chat(text):
    day = 0
    m_old = ""
    with open(text,"r",encoding="utf-8") as f:
        for line in f:
            m = re.search("20[0-9]{2}-[0-9]{2}-[0-9]{2}", line)
            if m is not None and m.group() != m_old:
                day = day + 1
                m_old = m.group()
    return day

##统计每个月份的聊天条数
def rank_of_chat_by_month_and_day(text):
    stat_month = collections.defaultdict(lambda:0)##字典使用时，会自动赋初值为0
    stat_day = collections.defaultdict(lambda:0)
    with open(text,"r",encoding="utf-8") as f:
        for line in f:
            m = re.search("20[0-9]{2}-[0-9]{2}-[0-9]{2}", line)
            if m is not None:
                month = m.group()[0:7]
                day = m.group()
                stat_day[day] = stat_day[day] + 1
                stat_month[month] = stat_month[month] + 1
    return stat_month, stat_day

##统计日聊天频率排行，前五名
def top5_of_chat_by_day(stat_day):
    return sorted(stat_day.items(),key = lambda x:x[1],reverse=True)[0:5]

##统计一天24h时段的聊天频率（每个时段多少聊天记录）
def stat_of_24hours(text):
    stat = [0]*24
    with open(text,"r",encoding="utf-8") as f:
        for line in f:
            m = re.search("[0-9]{2}:[0-9]{2}:[0-9]{2}",line)
            if m is not None:
                hour = int(m.group()[0:2])
                stat[hour] = stat[hour] + 1
    return stat

##表情包统计
def top10_of_face_expression(text):
    stat = collections.defaultdict(lambda:0)
    with open(text,"r",encoding="utf-8") as f:
        for line in f:
            m = re.findall("\[.{1,2}\]",line)##return type list for all match iterms
            if m is not []:
                for face in m:
                    stat[face] = stat[face] + 1
    return sorted(stat.items(),key = lambda x:x[1],reverse=True)[0:10]

##词云分析
def stat_of_wordcloud(text):
    clean_words = []
    remove_words = []

    with open(text,"r",encoding="utf-8") as f:
        data = f.read() ##读取整个文件
    
    pattern = re.compile("\t|-|:|[0-9]|\n|\r|\[|\]")
    data = re.sub(pattern, "", data)##去掉不想要的字符

    cut_words = jieba.cut(data, cut_all=False)##精确模式分词

    with open("stopwords.dat","r",encoding="utf-8") as sf:
        for line in sf:
            remove_words.append(line.strip())
    remove_words = remove_words + ["李秋华"," ","奸笑","捂脸","脸","年","是从","D","那种","有个","真","中","少","捂","呲","牙","太","远","里","完","号","\ue409","\ue412","点","高","好像","挺","回来","先"]

    for word in cut_words:
        if word not in remove_words:
            clean_words.append(word)
    
    word_counts = collections.Counter(clean_words)
    word_counts_top200 = word_counts.most_common(200)
    print(word_counts_top200)

    mask = np.array(Image.open('heart.jpg')) # 定义词频背景
    wc = wordcloud.WordCloud(   font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式 
                                background_color="black",
                                mask=mask, # 设置背景图 
                                max_words=200, # 最多显示词数 
                                max_font_size=150 # 字体最大值 
                            )
    wc.generate_from_frequencies(word_counts) # 从字典生成词云
    image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案 
    wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案 
    plt.imshow(wc) # 显示词云 
    plt.axis('off') # 关闭坐标轴 
    plt.show() # 显示图像

##图片数字显示格式
def func(pct, allvals, self_words):
     absolute = math.ceil(pct/100.*np.sum(allvals))
     return "{:.1f}%\n({:d} {:s})".format(pct, absolute, self_words)



###########################################################################################
# text = "wechat.txt"

###########################################################################################
# days, sec = len_of_chat(text)
# print("相识：%d天 %d小时 %d分钟 %d秒"%(days, sec/3600, sec%3600/60, sec%3600%60))


###########################################################################################
# stat_of_wordcloud(text)


###########################################################################################
##字典dic.items()后就会变成以元组为元素的列表
# lst_top10 = top10_of_face_expression(text)
# for k,v in lst_top10:
#    print("%s:%d"%(k,v))
# plt.figure(num="stat_of_face")
# values = [x[1] for x in lst_top10]
# plt.barh(range(10,0,-1),width=values,height=0.5,color="deepskyblue")
# plt.yticks([])
# plt.title("Statistics of WeChat expression",size=15,weight="bold")
# plt.show()


##########################################################################################
# clock_items = stat_of_24hours(text)
# for i,v in enumerate(clock_items):
#     print("%02d:%d"%(i,v))
# plt.figure(num="stat_of_24hours")
# labels = [str(x)+":00" for x in range(24)]
# values = clock_items
# ##使用range()作为x轴的标签排序（bar默认会按照标签字符串排序），使用tick_label去替换range定好顺序的标签名
# plt.bar(range(len(labels)), tick_label=labels, height=values,color="hotpink",width=0.5)
# plt.xlabel("time",fontsize=12)
# plt.ylabel("items",fontsize=12)
# plt.title("Statistics of WeChat items by 24 hours",size=15,weight="bold")
# plt.show()


#######################################################################################
# month,day = rank_of_chat_by_month_and_day(text)
# day_top5 = top5_of_chat_by_day(day)
# for k,v in month.items():
#    print("%s：%d"%(k,v))
# for i,k in enumerate(day_top5):
#    print("Top%d %s: %d"%(i,k[0],k[1]))
# plt.figure(num="stat_of_month")
# labels = [i[0] for i in month.items() ]
# values = [i[1] for i in month.items() ]
# x = range(len(labels))
# plt.plot(x, values, 'ro-')
# plt.xticks(x, labels)
# plt.xlabel("month",fontsize=12)
# plt.ylabel("items",fontsize=12)
# plt.title("Statistics of WeChat items by months",size=15,weight="bold")
# plt.show()

# plt.figure(num="stat_of_top5")
# labels = [i[0] for i in day_top5]
# values = [i[1] for i in day_top5]
# plt.bar(range(len(labels)), tick_label=labels, height=values, width=0.5, color="hotpink")
# plt.xlabel("date",fontsize=12)
# plt.ylabel("items",fontsize=12)
# plt.title("Statistic of WeChat items by days (Top 5)",size=15,weight="bold")
# plt.show()

###########################################################################################
##days:70
# chat_days = day_of_chat(text)
# print("days:%d"%(chat_days))
# plt.figure(num="stat_of_chat_days")
# legends=['Be a quiet pretty girl/boy.','I can I BB!']
# colors = ['deepskyblue','hotpink']
# fracs=[days - chat_days,chat_days]
# explodes=[0,0.02]
# wedges, texts, autotexts = plt.pie(x=fracs,colors=colors,autopct=lambda pct:func(pct, fracs, "days"),startangle=90,explode=explodes)#autopct显示百分比
# plt.legend(wedges,legends,loc="best")
# plt.title("Statistics of WeChat Days",size=15,weight="bold")
# plt.show()

########################################################################################
##统计每个人的聊天总条数，总字数，单条微信最大字数和索引处 me:1963,35260,130,6907  , you:2185,41784,1964,2209
# me_c, me_w, me_s, me_i = count_of_chat(text, "我")
# you_c, you_w, you_s, you_i = count_of_chat(text, "李秋华")
# print("me:%d,%d,%d,%d  , you:%d,%d,%d,%d "%(me_c, me_w, me_s, me_i, you_c, you_w, you_s, you_i))

# ##figure1 -> wechat items
# plt.figure(num="items")
# legends=['Crazy King','Lorraine']
# colors = ['deepskyblue','hotpink']
# fracs=[me_c,you_c]
# explodes=[0,0.02]
# wedges, texts, autotexts = plt.pie(x=fracs,colors=colors,autopct=lambda pct:func(pct, fracs, "items"),startangle=90,explode=explodes)#autopct显示百分比
# plt.legend(wedges,legends,loc="best")
# plt.title("Statistics of WeChat Items",size=15,weight="bold")

# ##figure2 -> wechat words
# plt.figure(num="words")
# fracs=[me_w,you_w]
# wedges, texts, autotexts = plt.pie(x=fracs,colors=colors,autopct=lambda pct:func(pct, fracs, "words"),startangle=90,explode=explodes)#autopct显示百分比
# plt.legend(wedges,legends,loc="best")
# plt.title("Statistics of WeChat Words",size=15,weight="bold")

# plt.setp(autotexts, size=12)
# plt.show()





            
