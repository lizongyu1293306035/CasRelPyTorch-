"""
@FileName：pred.py\n
@Description：\n
@Author：Li Zongyu\n
@Time：2023/3/16 11:07\n
"""
import torch
import os
import json
from transformers import BertTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tuple(triple_list):
    ret = []
    for triple in triple_list:
        ret.append((triple['subject'], triple['predicate'], triple['object']))
    return ret


def pred(data_iter, rel_vocab, config, model, output_path=None, output=True, h_bar=0.5, t_bar=0.5):
    """
    测试并计算各项指标
    :param data_iter: data迭代器
    :param rel_vocab: 关系词典
    :param config: 配置类
    :param model: 模型
    :param output: 是否输出(boolen)
    :param output_path: 输出文件路径（默认路径：config.result_dir + config.pred_result_save_name）
    :param h_bar:
    :param t_bar:
    :return: None
    """
    orders = ['subject', 'relation', 'object']
    tokenizer = BertTokenizer.from_pretrained(config.bert_name)
    for batch_x, batch_y in tqdm(data_iter):
        # batch_x: {"token_ids":tensor(int),
        #           'mask': tensor(bool),
        #           'sub_head': tensor(binary),
        #           'sub_tail': tensor(binary),
        #           'sub_heads': tensor(binary)
        #           }
        # batch_y: {'mask': tensor(bool),
        #           'sub_heads': tensor(binary),
        #           'sub_tails': tensor(binary),
        #           'obj_heads': tensor(binary),
        #           'obj_tails': tensor(binary),
        #           'triples': a data
        #           }
        with torch.no_grad():
            token_ids = batch_x['token_ids']
            mask = batch_x['mask']
            encoded_text = model.get_encoded_text(token_ids, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
            sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]
            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = ''.join(tokenizer.decode(token_ids[0][sub_head: sub_tail + 1]).split())
                    subjects.append((subject, sub_head, sub_tail))

            # !test
            print("subjects:", subjects)

            if subjects:
                triple_list = []
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                sub_head_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=device)
                sub_tail_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=device)
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[1]] = 1
                    sub_tail_mapping[subject_idx][0][subject[2]] = 1
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                 repeated_encoded_text)
                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]
                    obj_heads = torch.where(pred_obj_heads[subject_idx] > h_bar)
                    obj_tails = torch.where(pred_obj_tails[subject_idx] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = rel_vocab.to_word(int(rel_head))
                                obj = ''.join(tokenizer.decode(token_ids[0][obj_head: obj_tail + 1]).split())
                                triple_list.append((sub, rel, obj))
                                break

                triple_set = set()
                for s, r, o in triple_list:
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)

                # !test
                # print("pred_list:", pred_list)

            else:
                pred_list = []
            pred_triples = set(pred_list)
            if output:
                if not os.path.exists(config.result_dir):
                    os.makedirs(config.result_dir)
                if output_path is None:
                    output_path = os.path.join(config.result_dir, config.pred_result_save_name)
                fw = open(output_path, 'a')
                result = json.dumps(
                    {
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ]
                    }
                    , ensure_ascii=False)
                fw.write(result + '\n')

