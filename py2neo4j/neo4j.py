"""
@FileName：neo4j.py\n
@Description：\n
@Author：Li Zongyu\n
@Time：2023/3/16 17:14\n
"""
import json
from py2neo import Graph, Node, Relationship, NodeMatcher
from model.config import Config


def load_to_neo4j(con: Config):
    """
    连接neo4j数据库，并将关系生成图。

    :param con: 配置类
    :return: None
    """
    # 链接数据库
    # graph = Graph("http://localhost:7474", auth=("neo4j", "137920"))
    graph = Graph(con.localhost, auth=(con.neo_user, con.neo_password))
    # Open file
    fileHandler = open(con.result_dir + con.pred_result_save_name, "r")
    # Get list of all lines in file
    listOfLines = fileHandler.readlines()
    listOfDict = []
    # Close file
    fileHandler.close()
    # clean database
    graph.delete_all()
    for line in listOfLines:
        c = json.loads(line)
        listOfDict.append(c)
    for dict in listOfDict:
        for rel in dict["new"]:
            # type(rel) == dict. Such as:{'subject': '邓丽君', 'relation': '出生地', 'object': '中国台湾省云林县褒忠乡田洋村'}
            # sub = Node('sth', name=rel["subject"])  # 节点1
            # obj = Node('sth', name=rel["object"])   # 节点2
            # 检查两节点是否有相似的节点
            matcher1 = NodeMatcher(graph)
            nodelist1 = list(matcher1.match("sth", name=rel["subject"]))
            if len(nodelist1) == 0:
                # 如果节点1没有相似的节点，则创建新的节点1
                sub = Node('sth', name=rel["subject"])  # 节点1
            else:
                # 如果节点1有相似的节点，则找到该相似节点
                sub = nodelist1[0]
            matcher2 = NodeMatcher(graph)
            nodelist2 = list(matcher2.match("sth", name=rel["object"]))
            if len(nodelist2) == 0:
                # 如果节点1没有相似的节点，则创建新的节点1
                obj = Node('sth', name=rel["object"])  # 节点1
            else:
                # 如果节点2有相似的节点，则找到该相似节点
                obj = nodelist2[0]
            rel = Relationship(sub, rel["relation"], obj)
            graph.create(rel)  # 创建关系


if __name__ == '__main__':
    graph = Graph("http://localhost:7474", auth=("neo4j", "137920"))
    graph.delete_all()
