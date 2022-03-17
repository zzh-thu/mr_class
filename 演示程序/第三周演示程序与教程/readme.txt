1. 演示程序运行方法
在Anaconda Prompt命令行终端中，激活已安装PyTorch的conda 环境。
例如：
conda activate myenv

用cd命令进入程序所在目录(例如 Windows平台上自建目录c:\2022) ，再运行已放入该目录中的python演示程序：
cd c:\2022
python  week3-demo.py

该程序改编自https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html，改编部位为采用自定义的torch.nn.Module子类实现多层感知机。

2.  tutorial-3-MLP.ipynb
本教程包括三部分内容：多层感知机中常用的激活函数，利用张量计算底层方法实现多层感知机，以及利用PyTorch神经网络模块实现多层感知机。

