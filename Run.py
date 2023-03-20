import argparse
from fastNLP import Vocabulary
import torch
import torch.optim as optim
from fastNLP.io import DataBundle
from model.casRel import CasRel
from model.callback import MyCallBack
from model.data import load_data, get_data_iterator
from model.config import Config
from model.evaluate import metric
import torch.nn.functional as F
from fastNLP import Trainer, LossBase
from model.pred import pred
from py2neo4j.neo4j import load_to_neo4j
from tool.dataGeneration import to_json


class MyLoss(LossBase):
    def __init__(self):
        super(MyLoss, self).__init__()

    def get_loss(self, predict, target):
        mask = target['mask']

        def loss_fn(pred, gold, mask):
            pred = pred.squeeze(-1)
            loss = F.binary_cross_entropy(pred, gold, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            return loss

        return loss_fn(predict['sub_heads'], target['sub_heads'], mask) + \
               loss_fn(predict['sub_tails'], target['sub_tails'], mask) + \
               loss_fn(predict['obj_heads'], target['obj_heads'], mask) + \
               loss_fn(predict['obj_tails'], target['obj_tails'], mask)

    def __call__(self, pred_dict, target_dict, check=False):
        loss = self.get_loss(pred_dict, target_dict)
        return loss


def model_train(con: Config, data_bundle: DataBundle, rel_vocab: Vocabulary, model: CasRel):
    """
    模型训练。

    :param con: 配置类
    :param data_bundle: 一堆数据包类
    :param rel_vocab: 关系字典
    :param model: 初始模型
    :return: None
    """
    train_data = get_data_iterator(con, data_bundle.get_dataset('train'), rel_vocab)
    dev_data = get_data_iterator(con, data_bundle.get_dataset('dev'), rel_vocab, is_test=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr)
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer, batch_size=con.batch_size,
                      n_epochs=con.max_epoch, loss=MyLoss(), print_every=con.period, use_tqdm=True,
                      callbacks=MyCallBack(dev_data, rel_vocab, con))
    print("-" * 5 + "Begin Training" + "-" * 5)
    trainer.train()


def model_test(con: Config, data_bundle: DataBundle, rel_vocab: Vocabulary, model: CasRel):
    """
    模型测试。

    :param con: 配置类
    :param data_bundle: 一堆数据包类
    :param rel_vocab: 关系字典
    :param model: 模型
    :return: None
    """
    test_data = get_data_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)
    try:
        model.load_state_dict(torch.load(con.save_weights_dir + con.weights_save_name))
    except Exception as e:
        print(e)
    print("-" * 5 + "Begin Testing" + "-" * 5)
    metric(test_data, rel_vocab, con, model)


def model_predict(con: Config, data_bundle: DataBundle, rel_vocab: Vocabulary, model: CasRel):
    """
    使用训练后的模型进行预测。

    :param con: 配置类
    :param data_bundle: 存放所有数据集的类
    :param rel_vocab: 关系字典
    :param model: 未更新的预训练模型参数
    :return: None
    """
    pred_data = get_data_iterator(con, data_bundle.get_dataset('pred'), rel_vocab, is_test=True)
    try:
        model.load_state_dict(torch.load(con.save_weights_dir + con.weights_save_name))
    except Exception as e:
        print(e)
    print("-" * 5 + "Begin Predicting" + "-" * 5)
    pred(pred_data, rel_vocab, con, model, con.result_dir + con.pred_result_save_name)
    # 生成图数据库代码
    load_to_neo4j(con)


if __name__ == '__main__':
    # 生成随机数种子
    seed = 226
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--dataset', default='meikuang', type=str, help='define your own dataset names')
    # parser.add_argument('--dataset', default='baidu', type=str, help='define your own dataset names')
    parser.add_argument("--bert_name", default='bert-base-chinese', type=str, help='choose pretrained bert name')
    parser.add_argument('--bert_dim', default=768, type=int)
    args = parser.parse_args()
    con = Config(args)

    model = CasRel(con).to(device)
    to_json(con)
    data_bundle, rel_vocab = load_data(con.train_path, con.dev_path,
                                       con.test_path, con.pred_path,
                                       con.rel_path)

    # 训练模型
    # model_train(con, data_bundle, rel_vocab, model)
    #
    # 测试模型
    # model_test(con, data_bundle, rel_vocab, model)
    #
    # 预测模型
    model_predict(con, data_bundle, rel_vocab, model)


