"""
@FileName：dataGeneration.py\n
@Description：\n
@Author：Li Zongyu\n
@Time：2023/3/17 20:45\n
"""
import json

from model.config import Config


def to_json(con: Config):
    """
    从指定txt文件中读取文本，将文本转换为指定的输入格式，并将其保存到指定的json文件中。

    :param con: 配置类
    :return: None
    """
    with open(con.txt, 'r', encoding='utf-8') as f:
        text = f.read()
    # 将文本按照句号分成行
    lines = text.split('。')
    for i in range(len(lines)):
        # 将每行的句号加上，注意最后一行不需要加句号
        if i < len(lines) - 1:
            lines[i] += '。'
    # {"text": "个人生活李佳璇和导演路学长因拍摄《卡拉是条狗》而相识，2003年两人结婚", "spo_list": []}
    # 文本清空
    f = open(con.result_dir + con.pred_result_save_name, 'w').close()
    f = open(con.pred_path, 'w').close()
    for line in lines[0:-1]:
        dic = {"text": line, "spo_list": []}
        json_d = json.dumps(dic, ensure_ascii=False)
        with open(con.pred_path, 'a', encoding='utf-8') as file:
            file.write(json_d + "\n")
        file.close()
