# ABSA




##### 参考

- ABSA presentation [[ppt](https://www.iaria.org/conferences2016/filesHUSO16/OrpheeDeClercq_Keynote_ABSA.pdf)]
- 阿里云的商品评价解析 [[link](https://help.aliyun.com/document_detail/64231.html?spm=5176.12095382.1232858.4.739e3b24xUnvbZ)]



##### Task

- [ ] 提取观点
- [ ] 提取关键词
- [ ] EA分类
- [ ] polarity分类
- [ ] 观点过滤：文字噪音处理、虚假评论、水军、广告、不含观点、无意义文本
- [ ] negation 否定处理



##### SemEval ABSA

- NLP的 SemEval 论文合辑 [[ACL](https://www.aclweb.org/anthology/)]
- SemEval - 2014 - ABSA [[competition](http://alt.qcri.org/semeval2014/task4/)]
- SemEval - 2015 - ABSA [[competition](http://alt.qcri.org/semeval2015/task12/)]
- SemEval - 2016 - ABSA [[competition](http://alt.qcri.org/semeval2016/task5/)] [[guideline](http://alt.qcri.org/semeval2016/task5/data/uploads/absa2016_annotationguidelines.pdf)] [[paper](http://alt.qcri.org/semeval2016/task5/data/uploads/absa2016_annotationguidelines.pdf)]



##### 可参考的GitHub项目（消化完可随时移除）

顺便可以标识一下每个项目用的啥数据集，大部分是SemEval的

- [self data] [Unsupervised-Aspect-Extraction](https://github.com/ruidan/Unsupervised-Aspect-Extraction) 
- [SemEval-2016] [aspect-extraction](https://github.com/soujanyaporia/aspect-extraction) 
- [SemEval-2015] [AspectBasedSentimentAnalysis](https://github.com/yardstick17/AspectBasedSentimentAnalysis) 
- [SemEval-2016] [Review_aspect_extraction](https://github.com/yafangy/Review_aspect_extraction) 
- [SemEval-2014, 2016] [DE-CNN](https://github.com/howardhsu/DE-CNN) 
- [SemEval-2015] [Coupled-Multi-layer-Attentions](https://github.com/happywwy/Coupled-Multi-layer-Attentions) 
- [SemEval-2016 laptop] [mem_absa](https://github.com/ganeshjawahar/mem_absa) 
- [SemEval-2014] [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) 
- [SemEval-2014, 2016] [Attention_Based_LSTM_AspectBased_SA](https://github.com/gangeshwark/Attention_Based_LSTM_AspectBased_SA) 
- [SemEval-2014] [ABSA_Keras](https://github.com/AlexYangLi/ABSA_Keras) 
- [SemEval-2016] [ABSA](https://github.com/LingxB/ABSA/tree/master/Data/SemEval) 

 

##### paper

- Deep Learning for Aspect-Based Sentiment Analysis [[paper](https://cs224d.stanford.edu/reports/WangBo.pdf)]
- Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings [[paper](https://www.aclweb.org/anthology/D15-1168)]
- SemEval-2015 Task 12: Aspect Based Sentiment Analysis [[paper](https://www.aclweb.org/anthology/S15-2082)]
- SemEval-2016 Task 5: Aspect Based Sentiment Analysis [[paper](https://www.aclweb.org/anthology/S16-1002)]
- Encoding Conversation Context for Neural Keyphrase Extraction from Microblog Posts [[paper](https://ai.tencent.com/ailab/media/publications/naacl2018/Encoding_Conversation_Context_for_Neural_Keyphrase_Extraction_from_Microblog_Posts.pdf)]
- End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF [[paper](https://arxiv.org/pdf/1603.01354.pdf)]



##### 有趣想做的GitHub项目

- [ai-research-keyphrase-extraction](https://github.com/swisscom/ai-research-keyphrase-extraction)



##### 数据集

###### 中文

- AI-Challenge


###### 英文

- Amazon product data [[data](http://jmcauley.ucsd.edu/data/amazon/)]
- Web data: Amazon reviews[[data](https://snap.stanford.edu/data/web-Amazon.html)]





别人的书的目录，可以学习逻辑

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