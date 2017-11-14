
---
#  Source code for ADSH-AAAI2018 [Matlab Version]
---
## Introduction
### 1. Running Environment
Matlab 2016

[MatConvnet](http://www.vlfeat.org/matconvnet/)
### 2. Datasets
We use three datasets to perform our experiments, i.e., CIFAR-10, MS-COCO and NUS-WIDE. You can preprocess these datasets by yourself or download from the following links:

[CIFAR-10 MAT File](http://pan.baidu.com/s/1miMgd7q)

[MS-COCO MAT File]()

[NUS-WIDE MAT File]()

In addition, pretrained model can be download from the following links:

[VGG-F](http://pan.baidu.com/s/1slhusrF)

### 3. Run demo
First you need download (or prepross by yourself) coresponding data and pretrained model and put them in the "data" folder. Then complie and run setup.m to configure MatConvNet.
Then run ADSH_demo().
```matlab
ADSH_demo
```

### 4. Result
#### 4.1. Mean Average Precision (MAP).
<table>
    <tr>
        <td rowspan="2">PlatForm</td>    
        <td colspan="4">Code Length</td>
    </tr>
    <tr>
        <td >12 bits</td><td >24 bits</td> <td >32 bits</td><td >48 bits</td>  
    </tr>
    <tr>
        <td >CIFAR-10</td ><td >0.8939 </td> <td > 0.9243 </td><td > </td><td > </td>  
    </tr>
    <tr>
        <td >MS-COCO</td ><td > </td> <td > </td><td > </td> <td > </td>
    </tr>
    <tr>
        <td >NUS-WIDE</td ><td > </td> <td > </td><td > </td> <td > </td>
    </tr>
</table>
#### 4.2. Precision-Recall

![](./fig/PreRec.png)
#### 4.3. Training Loss on MS-COCO dataset.

![](./fig/MS-COCO-loss.png)
#### 4.4.  Hyper-parameters on MS-COCO dataset.
##### 4.4.1. $\gamma$

![](./fig/MS-COCO-gammas.png)
##### 4.4.2. $m$

![](./fig/MS-COCO-numsamples.png)


### 5. Please contact jiangqy@lamda.nju.edu.cn if you have any question.