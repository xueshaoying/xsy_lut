import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

# sampler used for meta-training
class meta_batchsampler(Sampler):
    
    def __init__(self,data_source,way,shots):

        self.way = way
        self.shots = shots
#创建一个类到其样本索引的字典class2id,该列表中包含了该类别下所有样本的索引值
        class2id = {}

#其输入参数data_source包含了元学习任务的所有数据。
#对于每个数据样本，由enumerate函数返回其在数据源中的索引i以及对应的类别标签class_id。
#然后，对于每个类别标签class_id，如果其不存在于class2id字典中，则将其作为key添加到字典中，并将其值初始化为空列表[]；
#如果已经存在，则直接添加索引i到对应的列表中。这样，class2id字典中每个key都对应着一个列表，该列表中包含了该类别下所有样本的索引值。
# 最后，将class2id字典存储到对象的self.class2id属性中，以便后续的mini-batch采样操作使用。
        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)
        self.class2id = class2id
#class2id
# 迭代函数__iter__(self)：在每次迭代时，先深度复制class2id到一个临时字典temp_class2id中，随机打乱每个类别的样本索引。
    def __iter__(self):
        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])
# 然后从temp_class2id中随机选择way个类别，按照每个类别shots个样本为一个 batch 进行采样，将采样到的样本索引存储到id_list中
        while len(temp_class2id) >= self.way:
            #按照way/shot采样后的样本索引到id_list
            id_list = []
            list_class_id = list(temp_class2id.keys())
#首先，统计每个类别中剩余的样本数，这里用列表推导式和numpy数组来实现。
#对于list_class_id列表中的每个类别标签class_id，通过temp_class2id[class_id]访问该类别下剩余样本的索引列表，
#并用len()函数获取其长度（即该类别下剩余样本数）,将其存储到一个numpy数组pcount中，该数组中第i个元素表示第i个类别的剩余样本数。
            #计算每个类别的数目
            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])
            batch_class_id = np.random.choice(list_class_id,size=self.way,replace=False,p=pcount/sum(pcount))
            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())
# 最后删除样本数不足的类别，然后yield返回id_list，即得到一个 mini-batch
            for class_id in batch_class_id:
                if len(temp_class2id[class_id])<sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list
# sampler used for meta-testing
class random_sampler(Sampler):

    def __init__(self,data_source,way,shot,query_shot=16,trial=1000):

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = 16

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot
        
        class2id = deepcopy(self.class2id)        
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []
 
            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])
                
            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot+query_shot)])

            yield id_list