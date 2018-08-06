# PyTorch Code for Generative Temporal Models with Memory (GTMM)

This repository implements the **Generative Temporal Models with Memory** as proposed in the paper:

**Gemici, Mevlana, et al. "Generative temporal models with memory." arXiv preprint arXiv:1702.04649 (2017).**

The code implements the following tasks so far:
1. **Perfect Recall Task** where given a sequence of *l* MNIST digits, the first *k* digits need to be recalled by the model. 
![perfect-recall-illustrations](https://user-images.githubusercontent.com/7714289/43701088-99f10146-9972-11e8-8225-ed7dbfd7ea92.png)

2. **Parity Recall Task** where given a sequence of *l* MNIST digits, the parities of the first *k* digits need to be recalled by the model and presented in the form of MNIST 0 or 1 digits. 0 corresponds to an odd digit and 1 corresponds to an even digit.\
![parity-recall-illustrations](https://user-images.githubusercontent.com/7714289/43701186-e52fdaa6-9972-11e8-84e7-c96bf615ad04.png)


## Known issues in this implementation
1. Parity recall tasks returns incorrect parity in some cases where it is hard to "classify" the MNIST digit image.
![parity-recall-misses](https://user-images.githubusercontent.com/7714289/43701320-5392deee-9973-11e8-94cc-3fdbc17703bc.png)
