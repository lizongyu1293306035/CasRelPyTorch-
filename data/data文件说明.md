# data文件说明：

data文件目录下存放着json格式的数据集，现阶段仅包含baidu数据集，数据集文件下共有6个文件：

- dev.json 是训练过程中的验证集；

- train.json 是训练集；

- test.json 是测试集；

- pred.json 是预测集;

- rel.json 是定义的关系字典

- text.txt 中存放着大量文本，对这些文本进行预处理后可以获得预测集合。文本的格式可以参照百度数据集中text.txt的格式进行书写。

  **注意：** 此项目由于本人项目经验有限，在数据集文件目录下本项目希望以上的六个文件都必须存在，且除了pred.json文件外其他文件均不为空。

