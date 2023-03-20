"""
@FileName：demo.py\n
@Description：\n
@Author：Li Zongyu\n
@Time：2023/3/15 17:54\n
"""
# 该python文件没用

import os
import torch

import json
from fastNLP import TorchLoaderIter, DataSet, Vocabulary, Sampler
from fastNLP.io import JsonLoader
from fastNLP.modules import BertModel
from torch import nn as nn
from transformers import BertModel


def b():
    str1 = "1"
    str2 = "2"
    return str1, str2;


def a() -> (str, str):
    str1 = "1"
    str2 = "2"
    return str1, str2;


if __name__ == '__main__':
    s1, s2 = a()
    print(s1, " ", s2)
    s1, s2 = b()
    print(s1, " ", s2)

# class Net(nn.Module):
#     def __init__(self):
#         # nn.Module子类的函数必须在构造函数中执行父类的构造函数
#         super(Net, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-chinese')
#
#
# net1 = Net()
# net2 = Net()
# for parameters in net1.parameters():
#     print(parameters)
#     break
#
# for parameters in net2.parameters():
#     print(parameters)
#     break
# class MyClass:
#     def __init__(self, num, step, start=0):
#         self.num = num
#         self.step = step
#         self.start = start
#
#     # 用于产生列表
#     def numlist(self):
#         return list(range(self.num))
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         numlist = self.numlist()
#         if self.start < len(numlist):
#             numsplit = numlist[self.start:(self.start + self.step)]
#             self.start += self.step
#             return numsplit
#         else:
#             raise StopIteration
#
#
# myiter = MyClass(num=20, step=5)
# for x in myiter:
#     print("o:", x)


#
# def load_data(train_path, dev_path, test_path, rel_dict_path):
#     paths = {'train': train_path, 'dev': dev_path, 'test': test_path}
#     loader = JsonLoader({"text": "text", "spo_list": "spo_list"})
#     data_bundle = loader.load(paths)
#     id2rel = json.load(open(rel_dict_path, encoding="utf-8"))
#     rel_vocab = Vocabulary(unknown=None, padding=None)
#     rel_vocab.add_word_lst(list(id2rel.values()))
#     return data_bundle, rel_vocab
#
#
# if __name__ == '__main__':
#     dataset = "baidu"
#     train_path = 'data/' + dataset + '/train.json'
#     test_path = 'data/' + dataset + '/test.json'
#     dev_path = 'data/' + dataset + '/dev.json'
#     rel_dict_path = 'data/' + dataset + '/rel.json'
#     data_bundle, rel_vocab = load_data(train_path, dev_path, test_path, rel_dict_path)
#     print("data_bundle:", data_bundle)
#     print("rel_vocab:", rel_vocab)


#
# result_dir = "saved_weights/baidu/"
# model_name = "model.pt"
#
# path = result_dir + model_name
# print(path)
# if not os.path.exists(path):
#     os.makedirs(result_dir)
# x = torch.tensor([0, 1, 2, 3, 4])
# torch.save(x, path)


# {"text": "7、刘晓庆1970年毕业于四川音乐学院附中，1975年走上银幕",
#  "spo_list": [{"predicate": "毕业院校",
#                "object_type": "学校",
#                "subject_type": "人物",
#                "object": "四川音乐学院",
#                "subject": "刘晓庆"}]
#  }
#
# {"text": "内容简介刘洪星编著的《考研高等代数辅导――精选名校真题》是数学类专业考研复习指导书",
#  "spo_list": [{"predicate": "作者",
#                "object_type": "人物",
#                "subject_type": "图书作品",
#                "object": "刘洪星",
#                "subject": "考研高等代数辅导"}]
#  }
# Save to file
# {"text": "白百何的处女座是《与青春有关的日子》，合作的演员是佟大为、陈羽凡",
#  "spo_list": [{"predicate": "主演",
#                "object_type": "人物",
#                "subject_type": "影视作品",
#                "object": "白百何",
#                "subject": "与青春有关的日子"},
#               {"predicate": "主演",
#                "object_type": "人物",
#                "subject_type": "影视作品",
#                "object": "陈羽凡",
#                "subject": "与青春有关的日子"},
#               {"predicate": "主演",
#                "object_type": "人物",
#                "subject_type": "影视作品",
#                "object": "佟大为",
#                "subject": "与青春有关的日子"}]
#  }
# {"triple_list_gold": [{"subject": "滴答",
#                        "relation": "歌手",
#                        "object": "侃侃"}],
#  "triple_list_pred": [{"subject": "北京爱情故事》的播出，歌手侃侃的在剧中演唱的插曲《滴答", "relation": "歌手", "object": "李晨"},
#                       {"subject": "滴答", "relation": "歌手", "object": "李晨"},
#                       {"subject": "滴答", "relation": "歌手", "object": "侃侃的在剧中演唱的插曲《滴答》在很短时间就在全国范围内走红，她更是凭借这首歌的超高人气与演员李晨"}],
#  "new": [{"subject": "北京爱情故事》的播出，歌手侃侃的在剧中演唱的插曲《滴答", "relation": "歌手", "object": "李晨"},
#          {"subject": "滴答", "relation": "歌手", "object": "李晨"},
#          {"subject": "滴答", "relation": "歌手", "object": "侃侃的在剧中演唱的插曲《滴答》在很短时间就在全国范围内走红，她更是凭借这首歌的超高人气与演员李晨"}],
#  "lack": [{"subject": "滴答", "relation": "歌手", "object": "侃侃"}]
#  }
# {"text": "新加坡总统陈庆炎：虽然邵逸夫先生后半生定居中国香港，但实际上他也为新加坡作出了很多贡献",
#  "spo_list": [{"predicate": "国籍",
#                "object_type": "国家",
#                "subject_type": "人物",
#                "object": "新加坡",
#                "subject": "陈庆炎"}]
#  }

# {"triple_list_gold": [{"subject": "酒店财务部精细化管理与服务规范（第2版）",
#                        "relation": "作者",
#                        "object": "蔡升桂"},
#                       {"subject": "酒店财务部精细化管理与服务规范（第2版）",
#                        "relation": "出版社",
#                        "object": "人民邮电"}],
#  "triple_list_pred": [{"subject": "酒店财务部精细化管理与服务规范",
#                        "relation": "出版社",
#                        "object": "人民邮电"}],
#  "new": [{"subject": "酒店财务部精细化管理与服务规范",
#           "relation": "出版社",
#           "object": "人民邮电"}],
#  "lack": [{"subject": "酒店财务部精细化管理与服务规范（第2版）",
#            "relation": "作者",
#            "object": "蔡升桂"},
#           {"subject": "酒店财务部精细化管理与服务规范（第2版）",
#            "relation": "出版社",
#            "object": "人民邮电"}]
#  }

# {"text": "《酒店财务部精细化管理与服务规范（第2版）》是2011年人民邮电出版社出版的图书，作者是蔡升桂",
#  "spo_list": [{"predicate": "作者",
#                "object": "蔡升桂",
#                "subject": "酒店财务部精细化管理与服务规范（第2版）"},
#               {"predicate": "出版社",
#                "object": "人民邮电",
#                "subject": "酒店财务部精细化管理与服务规范（第2版）"}]
#  }