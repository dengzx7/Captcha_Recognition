# Captcha_Recognition
SYSU Deep Learning Research Project<br>
基于AlexNet模型、tensorflow框架实现的验证码识别<br>
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/AlexNet.png" width="750">
</div>

## 项目概览
本项目按照以下步骤来实现验证码识别，训练50 Epochs后最终在测试集上的最高预测精度达到了 **97.5%**
* 爬取数据
* 数据预处理
* 训练模型与预测

## 项目环境
* 编程语言：python 3.6.4
* 学习框架：tensoflow 1.6.0
* 操作系统：Windows 10

## 运行方式
### 1. 爬取数据
**注：该步骤在项目中已完成，可以直接跳过该步骤<br>**
运行cmd指令，使用python requests爬取中山大学本科教务系统登陆界面的验证码
```
> python generate_captcha.py
```
一共将爬取920张图片，其中800张作为训练集，剩余的120张作为测试集，需要手动为这些验证码图片标注label；<br>
在[Captcha_Recognition/dataset/images](https://github.com/dengzx7/Captcha_Recognition/tree/master/dataset/images)
目录下有爬取好的数据集；<br>
在[Captcha_Recognition/dataset/labels](https://github.com/dengzx7/Captcha_Recognition/tree/master/dataset/labels)
目录下有对应标注好的标签，labels.txt不区分大小写，labels2.txt区分大小写，在本项目中使用了labels2.txt，下图展示了部分数据集<br>
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/%E9%AA%8C%E8%AF%81%E7%A0%81%E6%95%B0%E6%8D%AE%E9%9B%86.png" width="500">
</div>

### 2. 数据预处理
**注：该步骤在项目中已完成，可以直接跳过该步骤<br>**
若想重做该步骤，需要先将Captcha_Recognition/dataset目录下的以下文件删除
* preprocess
* segmatation
* test_file_labels.json
* train_file_labels.json

再运行cmd指令，数据预处理就完成了
```
> python preprocess_captcha.py
```
预处理验证码图片，包含以下四个步骤
* 去除干扰线
* 灰度化和二值化
* 去噪
* 图像分割

下图展示了一张原始验证码图片的预处理过程
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/%E9%A2%84%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.png" width="500">
</div>

图像分割过程是将一张包含四个字母或数字的验证码均分成四张图片，每一张分割后的图片包含一个字母或数字，AlexNet模型将对每一张分割后的图片进行训练，测试时先将测试集验证码分割为四张图片，若分割后的四张图片都预测正确，才算预测正确，分割后的部分结果如下图所示

<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/%E5%88%86%E5%89%B2%E5%90%8E%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86.png" width="550">
</div>

### 3. 训练模型与预测
运行cmd指令，开始模型训练和预测
```
> python train.py
```
在Captcha_Recognition/record目录下会生成record.txt文件和predict.txt文件，分别记录训练过程中的损失值cost、训练集精度和测试集精度，每10 Epochs会在Captcha_Recognition/alexnet_param目录下保存模型参数，训练50 Epochs后最终ALexNet模型在测试集上的最高预测精度达到了 **97.5%**
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/result.png" width="500">
</div>
