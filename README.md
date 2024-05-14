With the development of computer vision(CV), the technology of expression recognition has been improved by  intelligent recognition of human face images. At the same time, mood detection using EEG signals is also being developed. The multi-layer network algorithm has a good performance in CV  and EEG emotion recognition. However, the multi-layer network algorithm leads to high computing power and energy  consumption in the calculation process. Therefore, this paper proposes a lightweight Multi-modal algorithm for both CV  and EEG emotion recognition: Multi-modal decision fusion  neural network (MDFNN). The innovative points of this algorithm are as follows. The frequency spectrum features were extracted from EEG by the power spectral density(PSD), and then the features were classified into emotions by Gated Recurrent Unit(GRU) with attention mechanism. The  sequential characteristic of CV and the sequential information  of EEG signal are fused for multi-mode decision making. In  this paper, lightweight residual network and depth-separable  convolution are used as CV sub-classifier, and a single-layer GRU with attention mechanism is used as EEG sub-classifier. Then, the performance of these two sub-classifiers is  integrated through enumeration or Adaboost.