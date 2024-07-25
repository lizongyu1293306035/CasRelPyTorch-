# 写在前面

​	本人为完成研究生阶段的小作业，找到了源项目[CasRelPyTorch](https://github.com/Onion12138/CasRelPyTorch)。本项目修改自作者[Onion12138](https://github.com/Onion12138) 在2022年9月19日上传至Github的项目[CasRelPyTorch](https://github.com/Onion12138/CasRelPyTorch) ，本项目根据本人的需求修改部分代码以及增加了一些功能。如有任何侵权行为，请联系作者进行删除。

# 修改与增加的具体内容

​	作业的大概要求就是基于深度学习的方法完成实体关系抽取任务，并将抽取到的关系实体三元组输入到图数据库中（neo4j数据库）。

​	主要修改和增加功能的工作如下：

	- 修改了部分源码，在Run.py 中留出了模型训练、测试以及预测方法。
	- 对部分主要的代码添加了注释等，尽量地提高代码可读性。
	- 添加了模型预测方法。
	- 增加了将文本（.txt文件）转换成模型输入数据格式（.json文件）代码dataGeneration.py。
	- 增加了模型预测结果输出到图数据库neo4j的功能。
附百度数据集（源项目自带的数据集）训练的模型参数文件model.pt 如下：
链接：[https://pan.baidu.com/s/1lTOU3tUMFU-rKk24phzi2w?pwd=evxg ](https://pan.baidu.com/s/1bBQnTyh_M_PZrqwt7AQrCg?pwd=i4so)
提取码：i4so

# CasRel Model Pytorch reimplement 3
The code is the PyTorch reimplement of the paper "A Novel Cascade Binary Tagging Framework for Relational Triple Extraction" ACL2020. 
The [official code](https://github.com/weizhepei/CasRel) was written in keras. 

I have encountered a lot of troubles with the keras version, so I decided to rewrite the code in PyTorch.
# Introduction
I followed the previous work of [longlongman](https://github.com/longlongman/CasRel-pytorch-reimplement) 
and [JuliaSun623](https://github.com/JuliaSun623/CasRel_fastNLP).

So I have to express sincere thanks to them.

I made some changes in order to better apply to the Chinese Dataset.
The changes I have made are listed:
- I changed the tokenizer from HBTokenizer to BertTokenizer, so Chinese sentences are tokenized by single character.
  (Note that you don't need to worry about keras)
- I substituted the original pretrained model with 'bert-base-chinese'.
- I used fastNLP to build the datasets.
- I changed the encoding and decoding methods in order to fit the Chinese Dataset.
- I reconstruct the structure for readability.
# Requirements
- torch==1.8.0+cu111
- transformers==4.3.3
- fastNLP==0.6.0
- tqdm==4.59.0
- numpy==1.20.1
# Dataset
I preprocessed the open-source dataset from Baidu. I did some cleaning, so the data given have 18 relation types. 
Some noisy data are eliminated.

The data are in form of json. Take one as an example:
```json
{
    "text": "陶喆的一首《好好说再见》推荐给大家，希望你们能够喜欢",
    "spo_list": [
        {
            "predicate": "歌手",
            "object_type": "人物",
            "subject_type": "歌曲",
            "object": "陶喆",
            "subject": "好好说再见"
        }
    ]
}
```
In fact the field object_type and subject_type are not used.

If you have your own data, you can organize your data in the same format.
# Usage
1、cmd命令行使用方式（原文的使用方式）
```
python Run.py [--arg]
```
[--arg]：可选参数，例如：
```
python Run.py --dataset=baidu
```

I have already set the default value of the model, but you can still set your own configuration in model/config.py

2、编译器中运行使用方式
	此项目也可以在编译器中运行，在Run.py中给出了模型训练、模型测试以及模型预测三个功能函数，使用时只需修改

```
parser.add_argument('--dataset', default='baidu', type=str, help='define your own dataset names')
```
将上述代码中的default参数改写为自己的数据集即可。
# Results
The best F1 score on test data is 0.78 with a precision of 0.80 and recall of 0.76.

It is to my expectation although it may not reach its utmost.

I have also trained the [SpERT](https://github.com/lavis-nlp/spert) model, 
and CasRel turns out to perform better. 
More experiments need to be carried out since there are slight differences in both criterion and datasets.

# Experiences
- Learning rate 1e-5 seems a good choice. If you change the learning rate, the model will be dramatically affected.
- It shows little improvement when I substitute BERT with RoBERTa.
- It is crucial to shuffle the datasets in order to avoid overfitting. 



# CasRelPyTorch-
