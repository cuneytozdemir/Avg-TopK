**Avg-topk: A new pooling method for convolutional neural network**

This repository contains the implementation of the Avg-TopK pooling method proposed by Özdemir (2023) for Convolutional Neural Networks (CNNs). The implementation of the Avg-TopK pooling layer can be found in the `AvgTopKPooling.py` file.

Pooling layers are integral components in CNN architectures, responsible for reducing the size of feature maps while preserving essential information. Traditional pooling methods like maximum and average pooling have limitations in preserving dominant features effectively. To address these limitations, Özdemir (2023) introduced the Avg-TopK pooling method, which calculates the weighted average of dominant features and selects the top K pixels with the highest interaction for averaging.

The implementation in this repository includes the Avg-TopK pooling layer integrated into popular deep learning frameworks. Experimental results, as reported in Özdemir's study, demonstrate that the Avg-TopK pooling method achieves significantly higher image classification accuracy compared to conventional pooling methods.

For more details on the Avg-TopK pooling method and its implementation, please refer to the corresponding paper:

Özdemir, C. (2023). Avg-topk: A new pooling method for convolutional neural networks. Expert Systems with Applications, 119892.


https://www.sciencedirect.com/science/article/pii/S0957417423003937
