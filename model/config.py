import json


class Config(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_name = args.bert_name
        self.bert_dim = args.bert_dim

        self.train_path = 'data/' + self.dataset + '/train.json'
        self.test_path = 'data/' + self.dataset + '/test.json'
        self.pred_path = 'data/' + self.dataset + '/pred.json'
        # 验证集
        self.dev_path = 'data/' + self.dataset + '/dev.json'
        self.rel_path = 'data/' + self.dataset + '/rel.json'
        self.num_relations = len(json.load(open(self.rel_path, 'r', encoding="utf-8")))

        self.save_weights_dir = 'saved_weights/' + self.dataset + '/'
        self.save_logs_dir = 'saved_logs/' + self.dataset + '/'
        self.result_dir = 'results/' + self.dataset + '/'

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.test_result_save_name = 'test_result.json'
        self.pred_result_save_name = 'pred_result.json'

        # neo4j数据库配置
        self.neo_user = "neo4j"
        self.neo_password = "137920"
        self.localhost = "http://localhost:7474"

        # 煤矿数据
        self.txt = 'data/' + self.dataset + '/text.txt'
        # self.meikuang_pred_path = 'data/meikuang/pred.json'
        # self.meikuang_init_txt = "data/meikuang/text.txt"
        # self.meikuang_rel = "data/meikuang/rel.json"
        # self.meikuang_pred_result = "results/meikuang/"
        # self.meikuang_pred_result_save_name = 'pred_result.json'
        #
        # self.meikuang_pred_path = 'data/' + self.dataset + '/pred.json'
        #
        # self.meikuang_rel = 'data/' + self.dataset + '/rel.json'
        # self.meikuang_pred_result = 'results/' + self.dataset + '/'
        # self.meikuang_pred_result_save_name = 'pred_result.json'

