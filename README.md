[AI Benchmark Alpha](http://ai-benchmark.com/alpha) is an open source python library for evaluating AI performance of various hardware platforms, including CPUs, GPUs and TPUs. The benchmark is relying on [TensorFlow](https://www.tensorflow.org) machine learning library, and is providing a lightweight and accurate solution for assessing inference and training speed for key Deep Learning models.</br></br>

In total, AI Benchmark consists of <b>42 tests</b> and <b>19 sections</b> provided below:</br>

1. MobileNet-V2&nbsp; `[classification]`
2. Inception-V3&nbsp; `[classification]`
3. Inception-V4&nbsp; `[classification]`
4. Inception-ResNet-V2&nbsp; `[classification]`
5. ResNet-V2-50&nbsp; `[classification]`
6. ResNet-V2-152&nbsp; `[classification]`
7. VGG-16&nbsp; `[classification]`
8. SRCNN 9-5-5&nbsp; `[image-to-image mapping]`
9. VGG-19&nbsp; `[image-to-image mapping]`
10. ResNet-SRGAN&nbsp; `[image-to-image mapping]`
11. ResNet-DPED&nbsp; `[image-to-image mapping]`
12. U-Net&nbsp; `[image-to-image mapping]`
13. Nvidia-SPADE&nbsp; `[image-to-image mapping]`
14. ICNet&nbsp; `[image segmentation]`
15. PSPNet&nbsp; `[image segmentation]`
16. DeepLab&nbsp; `[image segmentation]`
17. Pixel-RNN&nbsp; `[inpainting]`
18. LSTM&nbsp; `[sentence sentiment analysis]`
19. GNMT&nbsp; `[text translation]`

For more information and results, please visit the project website: [http://ai-benchmark.com/alpha](http://ai-benchmark.com/alpha)</br></br>

#### Run Instructions </br>

Only for training:
```sudo docker run --gpus all -it odyssee/ai-benchmark:latest /bin/sh -c "./bin/ai-benchmark"```
