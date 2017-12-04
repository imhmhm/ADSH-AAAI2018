
---
#  Source code for ADSH-AAAI2018 [Pytorch Version]
---
## Introduction
### 1. Brief Introduction
This package contains the code for paper Asymmetric Deep Supervised Hashing on AAAI-2018. We only carry out experiment on CIFAR-10. And we utilize pre-trained ResNet-50 for feature learning rather CNN-F in our original paper. Please note that the results for paper is based on MatConvNet version.
### 2. Running Environment
```python
python3
pytorch
```
### 3. Datasets
We only carry out experiment on CIFAR-10 dataset. Please download matlab CIFAR-10 data and run DataPrepare.m and SaveFig.m to generate data. Or you can re-implement data_processing.py to loader data.
### 4. Run Demo
```python
python ADSH_CIFAR_10.py
```
### 5. Results
Mean Average Precision
<table>
    <tr>
        <td rowspan="2">Dataset</td>    
        <td colspan="4">Code Length</td>
    </tr>
    <tr>
        <td >12 bits</td><td >24 bits</td> <td >32 bits</td><td >48 bits</td>  
    </tr>
    <tr>
        <td >CIFAR-10</td ><td >0.9473 </td> <td > 0.9636 </td><td > 0.9632</td><td > 0.9605</td>  
    </tr>
</table>

### 6. Please contact jiangqy@lamda.nju.edu.cn if you have any question.