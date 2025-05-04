## ⚠️ Archived Repository Notice

**This repository has been archived** and is no longer maintained. It contains Python code for DeepMIMOv2 and DeepMIMOv3.

All future versions are now maintained in the [DeepMIMO Unified Repository](https://github.com/DeepMIMO/DeepMIMO)

DeepMIMOv4 is being developed in this repository and will be made public in May 2025. 

---

# DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications
This is a python code package of the DeepMIMO dataset generated using [Remcom Wireless InSite](http://www.remcom.com/wireless-insite) software. The [DeepMIMO dataset](https://deepmimo.net/) is a publicly available parameterized dataset published for deep learning applications in mmWave and massive MIMO systems.

This python code package is related to the following article: 
>Ahmed Alkhateeb, “[DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](https://arxiv.org/pdf/1902.06435.pdf),” in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019.
# Abstract of the Article
Machine learning tools are finding interesting applications in millimeter wave (mmWave) and massive MIMO systems. This is mainly thanks to their powerful capabilities in learning unknown models and tackling hard optimization problems. To advance the machine learning research in mmWave/massive MIMO, however, there is a need for a common dataset. This dataset can be used to evaluate the developed algorithms, reproduce the results, set *benchmarks*, and compare the different solutions. In this work, we introduce the DeepMIMO dataset, which is a generic dataset for mmWave/massive MIMO channels. The DeepMIMO dataset generation framework has two important features. First, the DeepMIMO channels are constructed based on accurate ray-tracing data obtained from Remcom Wireless InSite. The DeepMIMO channels, therefore, capture the dependence on the environment geometry/materials and transmitter/receiver locations, which is essential for several machine learning applications. Second, the DeepMIMO dataset is generic/parameterized as the researcher can adjust a set of system and channel parameters to tailor the generated DeepMIMO dataset for the target machine learning application. The DeepMIMO dataset can then be completely defined by the (i) the adopted ray-tracing scenario and (ii) the set of parameters, which enables the accurate definition and reproduction of the dataset. In this paper, an example DeepMIMO dataset is described based on an outdoor ray-tracing scenario of 18 base stations and more than one million users. The paper also shows how this dataset can be used in an example deep learning application of mmWave beam prediction.

# Installation & Dataset Generation
**To generate the dataset, please refer to [this website](https://deepmimo.net/versions/v2-python/) for the different dataset options and their generation steps.

If you have any questions regarding the code and used dataset, please contact [Ahmed Alkhateeb](https://www.aalkhateeb.net/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite both the original article and the Remcom Wireless InSite website:
> - A. Alkhateeb, “[DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](https://arxiv.org/pdf/1902.06435.pdf),” in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019.
> - Remcom, Wireless insite, “http://www.remcom.com/wireless-insite”
