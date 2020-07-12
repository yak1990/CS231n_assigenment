# 作业一
# 作业二

1. relu梯度，在0时应该是0.5，大于0是1，小于0是0
2. 网络中w初始化，在网络结构不深是，初始化的std小一些；在网络结构深的时候，随机化std大一些，免得梯度丢失
3. learning rate设置中，可以参考课件中的一个曲线图，根据loss的曲线判断lr设置大了还是小了
4. batchnorm中batchnorm_backward与batchnorm_backward_alt区别