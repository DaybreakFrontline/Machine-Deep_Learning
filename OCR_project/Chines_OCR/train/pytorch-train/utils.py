#!/usr/bin/python
# encoding: utf-8

import collections

import torch
import torch.nn as nn


class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + u'-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
        # print(self.dict)

    def encode(self, text, depth=0):
        """Support batch or single str."""
        length = []
        result = []
        for item in text:
            #FIXME
            #item = item.decode('utf-8', 'strict')
            length.append(len(item))
            # print(len(item))
            # print(item)
            for char in item:
                index = self.dict[char]
                result.append(index)

        text = result
        # print(text, length)
        # print(length)
        return (torch.IntTensor(text), torch.IntTensor(length))

        '''
        if isinstance(text, str):
            text = [self.dict[char.lower()] for char in text]
            length = [len(text)]

        if isinstance(text, str):
            text = [self.dict.get(char, 0) for char in text]
            length = [len(text)]
        # add for unicode python2
        # elif isinstance(text, unicode):
        #    text = [self.dict.get(char, self.dict[u'-']) for char in text]
        #    length = [len(text)]

        elif isinstance(text, collections.Iterable):
            length = [len(text)]
            # print("length = %s" % length)
            text = ''.join(str(v) for v in text)
            # print("text = %s" % text)
            text, _ = self.encode(text)

        if depth:
            return text, len(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        self.n_count += v.data.numel()
        # NOTE: not `+= v.sum()`, which will add a node in the compute graph,
        # which lead to memory leak
        self.sum += v.data.sum()

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img