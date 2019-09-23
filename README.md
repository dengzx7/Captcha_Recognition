# Captcha Recognition
Deep Learning Research Project (Smart Mobile networking and Computing Lab)<br>
基于Tensorflow框架、AlexNet模型和ResNet模型实现的验证码识别任务<br>

## 项目概览
本项目按照以下步骤来实现验证码识别，训练50 Epochs后，AlexNet在测试集上的最高预测精度达到了 **97.5%** ，ResNet在测试集上的最高预测精度达到了 **98%**
* 爬取数据 **Requests**
* 数据预处理 **Preprocess**
* 训练模型与预测 **Train**
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/%E6%A1%86%E6%9E%B6.PNG" width="750">
</div>

## 项目环境
* 编程语言：python 3.6.4
* 学习框架：tensoflow 1.6.0
* 操作系统：Windows 10

## 运行方式
### 一、爬取数据
**注：该步骤在项目中已完成，可以直接跳过该步骤<br>**
运行cmd指令，使用python requests爬取SYSU本科教务系统登陆界面的验证码
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

### 二、数据预处理
运行cmd指令，执行数据预处理
```
> python preprocess_captcha.py
```
执行后将在Captcha_Recognition/dataset目录下生成以下文件夹或文件
* preprocess
* segmatation
* test_file_labels.json
* train_file_labels.json

预处理验证码图片，包含以下四个步骤
* 去除干扰线
* 灰度化和二值化
* 去噪
* 图像分割

下图展示了一张原始验证码图片的预处理过程
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/%E9%A2%84%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.png" width="500">
</div>

图像分割过程是将一张包含四个字母或数字的验证码均分成四张图片，每一张分割后的图片包含一个字母或数字，分类模型将对每一张分割后的图片进行训练，测试时先将测试集验证码分割为四张图片，若分割后的四张图片都预测正确，才算预测正确，分割后的部分结果如下图所示

<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/%E5%88%86%E5%89%B2%E5%90%8E%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86.png" width="550">
</div>

### 三、训练模型与预测
在Captcha_Recognition/record目录下会生成record.txt文件和predict.txt文件，分别记录训练过程中的损失值loss、训练集精度和测试集精度，当程序结束时会在Captcha_Recognition/model_param目录下保存模型参数
### 3.1 AlexNet
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/AlexNet.png" width="750">
</div>

运行cmd指令，训练AlexNet模型
```
> python train.py --model alexnet
```

训练50 Epochs后最终ALexNet模型在测试集上的最高预测精度达到了 **97.5%**
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/alexnet_result.png" width="500">
</div>

### 3.2 ResNet(34-layer)
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/ResNet.png" width="500">
</div>

运行cmd指令，训练ResNet模型
```
> python train.py --model resnet
```

训练50 Epochs后最终ALexNet模型在测试集上的最高预测精度达到了 **98%**
<div align=center>
<img src="https://github.com/dengzx7/Captcha_Recognition/blob/master/images/resnet_result.png" width="500">
</div>

