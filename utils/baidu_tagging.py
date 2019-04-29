#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  aip  import  AipNlp
import  pandas  as  pd
import  time

"""  你的  APPID  AK  SK  """
APP_ID  =  '155934'
API_KEY  =  'PBW2w1dveS7x3YcKSZW0V7'
SECRET_KEY  =  'AOE75EWZqeI6kM7Kesq8i6FzQruDI'
client  =  AipNlp(APP_ID,  API_KEY,  SECRET_KEY)

# 请求文件
source_file  =  "请求文件路径"
source_df  =  pd.read_excel(source_file)


comments  =  []
neg_probs  =  []
pos_probs  =  []
confidences  =  []
sentiments  =  []
complete_count  =  0
#  请求错误统计
err_count  =  0
err_comment  =  []
start_time  =  time.time()
#  循环请求
i  =  0
while  i  <  len(source_df):
        comment  =  source_df["comment_content"][i]
        try:
                query_result  =  client.sentimentClassify(comment[:1024])
        except  Exception  as  e:
                print("query_result:{}".format(query_result))
                print("#######请求过程存在问题#######")
                err_count  +=  1
                err_comment.append(comment)
                i  +=  1
                continue
        try:
                result  =  query_result['items'][0]
                neg_prob  =  result['negative_prob']
                pos_prob  =  result['positive_prob']
                confidence  =  result['confidence']
                sentiment  =  result['sentiment']
        except  KeyError  as  e:
                print("#######请求QPS限制#######")
                print("i={}".format(i))
                continue
        i  +=  1
        comments.append(comment)
        neg_probs.append(neg_prob)
        pos_probs.append(pos_prob)
        confidences.append(confidence)
        sentiments.append(sentiment)
        complete_count  +=  1
        print("总共：{}条".format(len(source_df)))
        print("请求完成:  {}条".format(complete_count))
        print("完成进度：{}%".format(round(complete_count  /  len(source_df)  *  100,  2)))
        cost_mins  =  (time.time()  -  start_time)  /  60
        print("累计用时：{}分钟".format(round(cost_mins,  2)))
        avg_query_time  =  complete_count  /  cost_mins
        #  print("每条请求平均用时：{}".format(avg_query_time))
        left_mins  =  (len(source_df)  -  complete_count  -  err_count)  /  avg_query_time
        print("预计还需：{}分钟".format(round(left_mins,  2)))
        print("\n")

print("所有请求完成！")
print("请求总数量：{}".format(len(source_df)))
print("请求过程中存在问题的数量：{}".format(err_count))



#  保存结果

# 请求成功的结果保存
desti_df  =  pd.DataFrame()
desti_df["comment"]  =  comments
desti_df["neg_probs"]  =  neg_probs
desti_df["pos_probs"]  =  pos_probs
desti_df["confidences"]  =  confidences
desti_df["sentiments"]  =  sentiments
desti_file  =  "请求结果保存路径"
desti_df.to_excel(desti_file, engine='xlsxwriter')

# 请求失败的结果保存
err_df  =  pd.DataFrame()
err_file  =  "请求结果报错保存路径"
err_df["comment"] = err_comment
err_df.to_excel(err_file, engine='xlsxwriter') # 如果请求接口里有奇怪字符，保存文件时就使用, engine='xlsxwriter'