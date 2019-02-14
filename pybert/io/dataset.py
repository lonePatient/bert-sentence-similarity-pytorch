#encoding:utf-8
import csv
import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid   = guid  # 该样本的唯一ID
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    '''
    数据的feature集合
    '''
    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids   = input_ids   # tokens的索引
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id

class CreateDataset(Dataset):
    def __init__(self,data_path,max_seq_len,vocab_path,example_type,seed):
        self.seed    = seed
        self.max_seq_len  = max_seq_len
        self.example_type = example_type
        self.data_path  = data_path
        self.vocab_path = vocab_path
        self.reset()

    # 初始化
    def reset(self):
        # 加载语料库，这是pretrained Bert模型自带的
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_path)
        # 构建examples
        self.build_examples()

    # 读取数据集
    def read_data(self,quotechar = None):
        '''
        默认是以tab分割的数据
        :param quotechar:
        :return:
        '''
        lines = []
        with open(self.data_path,'r',encoding='utf-8') as fr:
            reader = csv.reader(fr,delimiter = '\t',quotechar = quotechar)
            for line in reader:
                lines.append(line)
        return lines

    # 构建数据examples
    def build_examples(self):
        lines = self.read_data()
        self.examples = []
        for i,line in enumerate(lines):
            guid = '%s-%d'%(self.example_type,i)
            label = line[0]
            text_a = line[1]
            text_b = line[2]
            example = InputExample(guid = guid,text_a = text_a,text_b=text_b,label= label)
            self.examples.append(example)
        del lines

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    # 将example转化为feature
    def build_features(self,example):
        '''
        # 对于两个句子:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        # 对于单个句子:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # type_ids:表示是第一个句子还是第二个句子
        '''
        #转化为token
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = self.tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        self.truncate_seq_pair(tokens_a,tokens_b,max_length = self.max_seq_len - 3)
        # 第一个句子
        # 句子首尾加入标示符
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)  # 对应type_ids

        # 第二个句子
        tokens += tokens_b + ['[SEP]']
        segment_ids += [1] * (len(tokens_b) + 1)

        # 将词转化为语料库中对应的id
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # 输入mask
        input_mask = [1] * len(input_ids)
        # padding，使用0进行填充
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        # 标签
        label_id = int(example.label)
        feature = InputFeature(input_ids = input_ids,input_mask = input_mask,
                               segment_ids = segment_ids,label_id = label_id)
        return feature

    def _preprocess(self,index):
        example = self.examples[index]
        feature = self.build_features(example)
        return np.array(feature.input_ids),np.array(feature.input_mask),\
               np.array(feature.segment_ids),np.array(feature.label_id)

    def __getitem__(self, index):
        return self._preprocess(index)

    def __len__(self):
        return len(self.examples)
