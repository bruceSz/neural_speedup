迭代： 
1 增加sampleOnnxBert.cpp
2 上述cpp中来自tensorrt 带的samples例子，主要修改：
  1> 修改默认initializeSampleParams 中onnxFilleName 为params.onnxFileName = "model-sim.onnx";
  2> 修改输入initializeSampleParams 中输出tensorName 为input_ids, 以及logits
  3> 修改processInput，使用默认testWordsId ： std::vector<int> testWordIds = {101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102};
  4> 输入输出时间统计
3 sampleonnx.log 是上述编译二进制输出，log 显示tensort engine推理（加上host -> device , device ->host）时间为1.449。