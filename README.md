# ABSA

Aspect Based Sentiment Analysis

虽说是基于观点的分析，但也是基于句子层的分析，因为需要按句子进行分析。




##### 概念参考

- ABSA refer presentation [[ppt](https://www.iaria.org/conferences2016/filesHUSO16/OrpheeDeClercq_Keynote_ABSA.pdf)]
- 阿里云的商品评价解析 [[link](https://help.aliyun.com/document_detail/64231.html?spm=5176.12095382.1232858.4.739e3b24xUnvbZ)]

| 参数名         | 值                                                           |
| -------------- | ------------------------------------------------------------ |
| textPolarity   | 整条文本情感极性：正、中、负，text字段输入非法时返回-100     |
| textIntensity  | 整条文本情感程度(取值范围[-1,1]，越大代表越正向，越小代表越负向，接近0代表中性) |
| aspectItem     | 属性情感列表，每个元素是一个json字段                         |
| aspectCategory | 属性类别                                                     |
| aspectIndex    | 属性词所在的起始位置，终结位置                               |
| aspectTerm     | 属性词                                                       |
| opinionTerm    | 情感词                                                       |
| aspectPolarity | 属性片段极性（正、中、负）                                   |



##### Task Process

1. 按句 提取 属性词
2. 按句 提取 情感词
3. 属性词所在起始位置，终止位置
4. 属性词 -> EA分类
5. 情感词 -> 极性分类
6. 整条文本的感情极性（正、负、中） 及其概率值



##### Done Tasks

根据现有数据集，实际完成的任务

- [x] 按句进行 EA 分类
- [x] 按句进行情感极性分析



##### To do

- [ ] 观点过滤：文字噪音处理、虚假评论、水军、广告、不含观点、无意义文本
- [ ] negation 否定处理



##### SemEval ABSA

- NLP的 SemEval 论文合辑 [[ACL](https://www.aclweb.org/anthology/)]
- SemEval - 2014 - ABSA [[competition](http://alt.qcri.org/semeval2014/task4/)] [[data](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)] 
- SemEval - 2015 - ABSA [[competition](http://alt.qcri.org/semeval2015/task12/)] [[data](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)] [[paper](https://www.aclweb.org/anthology/S15-2082)] 
- SemEval - 2016 - ABSA [[competition](http://alt.qcri.org/semeval2016/task5/)] [[data](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)] [[guideline](http://alt.qcri.org/semeval2016/task5/data/uploads/absa2016_annotationguidelines.pdf)] [[paper](https://www.aclweb.org/anthology/S16-1002)]
- bonus: CodaLab Competitions [[intro](https://www.hse.ru/data/2017/05/31/1171931089/CodaLabCompetitions.pdf)] 



##### 可参考的GitHub项目

数据集基本都基于 2014-2016 SemEval 比赛

- [data: self data] [Unsupervised-Aspect-Extraction](https://github.com/ruidan/Unsupervised-Aspect-Extraction) 
- [data: SemEval-2016] [aspect-extraction](https://github.com/soujanyaporia/aspect-extraction) 
- [data: SemEval-2015] [AspectBasedSentimentAnalysis](https://github.com/yardstick17/AspectBasedSentimentAnalysis) 跑了下这个项目，其中结合了语法分析和机器学习，按照语法规则抽取的属性词。代码嵌套逻辑比较强，不建议套用。
- [data: SemEval-2016] [Review_aspect_extraction](https://github.com/yafangy/Review_aspect_extraction) 
- [data: SemEval-2014, 2016] [DE-CNN](https://github.com/howardhsu/DE-CNN) 
- [data: SemEval-2015] [Coupled-Multi-layer-Attentions](https://github.com/happywwy/Coupled-Multi-layer-Attentions) 
- [data: SemEval-2016 laptop] [mem_absa](https://github.com/ganeshjawahar/mem_absa) 
- [data: SemEval-2014] [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) 
- [data: SemEval-2014, 2016] [Attention_Based_LSTM_AspectBased_SA](https://github.com/gangeshwark/Attention_Based_LSTM_AspectBased_SA) 
- [data: SemEval-2014] [ABSA_Keras](https://github.com/AlexYangLi/ABSA_Keras) 利用了tensorflow hub，适用hub时出现了版本问题未跑通。
- [data: SemEval-2016] [ABSA](https://github.com/LingxB/ABSA/tree/master/Data/SemEval) 

 

##### paper

- Deep Learning for Aspect-Based Sentiment Analysis [[paper](https://cs224d.stanford.edu/reports/WangBo.pdf)]
- Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings [[paper](https://www.aclweb.org/anthology/D15-1168)]
- Encoding Conversation Context for Neural Keyphrase Extraction from Microblog Posts [[paper](https://ai.tencent.com/ailab/media/publications/naacl2018/Encoding_Conversation_Context_for_Neural_Keyphrase_Extraction_from_Microblog_Posts.pdf)]
- End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF [[paper](https://arxiv.org/pdf/1603.01354.pdf)]
- [2012] 用户评论中的标签抽取以及排序 [[paper](http://lipiji.com/docs/li2011opinion.pdf)] 



##### 数据集

###### 中文

- AI-Challenge [[data](https://drive.google.com/file/d/1OInXRx_OmIJgK3ZdoFZnmqUi0rGfOaQo/view)] 
- SemEval ABSA 2016 [[data](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)] 


###### 英文

- Amazon product data [[data](http://jmcauley.ucsd.edu/data/amazon/)]
- Web data: Amazon reviews [[data](https://snap.stanford.edu/data/web-Amazon.html)]
- Amazon Fine Food Reviews [[kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews)]
- SemEval ABSA



#### 优化方向

##### 字/词/句 文本嵌入Embedding

###### 中文

- Chinese Word Vectors [[github](https://github.com/Embedding/Chinese-Word-Vectors)] 
- nlp_chinese_corpus [[github](https://github.com/brightmart/nlp_chinese_corpus)] 
- 泛化语料、专业语料、向量化时，如何整合，还是两者独立向量化





ABSA书的目录，可以学习逻辑

#### ABSA Book Outline

1. Introduction
2. Aspect-Based Sentiment Analysis (ABSA)
   - 2.1. The three tasks of ABSA
   - 2.2. Domain and benchmark datasets
   - 2.3. Previous approaches to ABSA tasks
   - 2.4. Evaluation measures of ABSA tasks
3. Deep Learning for ABSA
   - 3.1. Multiple layers of DNN
   - 3.2. Initialization of input vectors
     - 3.2.1. Word embeddings vectors
     - 3.2.2. Featuring vectors
     - 3.2.3. Part-Of-Speech (POS) and chunk tags
     - 3.2.4. Commonsense knowledge
   - 3.3. Training process of DNNs
   - 3.4. Convolutional Neural Network Model (CNN)
     - 3.4.1. Architecture
     - 3.4.2. Application in consumer review domain
   - 3.5. Recurrent Neural Network Models (RNN)
     - 3.5.1. Computation of RNN models
     - 3.5.2. Bidirectional RNN
     - 3.5.3. Attention mechanism and memory networks
     - 3.5.4. Application in the consumer review domain
     - 3.5.5. Application in targeted sentiment analysis
   - 3.6. Recursive Neural Network Model (RecNN)
     - 3.6.1. Architecture
     - 3.6.2. Application
   - 3.7. Hybrid models
4. Comparison of performance on benchmark datasets
   - 4.1. Opinion target extraction
   - 4.2. Aspect category detection
   - 4.3. Sentiment polarity of aspect-based consumer reviews
   - 4.4. Sentiment polarity of targeted text
5. Challenges
   - 5.1. Domain adaptation
   - 5.2. Multilingual application
   - 5.3. Technical requirements
   - 5.4. Linguistic complications
6. Conclusion
7. Appendix: List of Abbreviations
8. References