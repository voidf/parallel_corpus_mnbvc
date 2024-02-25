import string
import jieba
import re
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn


punctuation_set = {
        '，', ',',
        '。', '.',
        '：', ',',
        '？', '?',
        '！', '!',
        '；', ';',
        '“', '"',
        '”', '"',
        '（', '(',
        '）', ')'
}

def filter_duplicated_whitespaces(src: str) -> str:
    """去噪：
        1. 如果换行符跟其它空格字符相连，这些字符替换成换行符
        2. 连续出现空格字符的，替换成其中一个空格字符"""
    whitespaces_set = set(string.whitespace.replace('\n', ''))
    buf = []
    newline = 0
    space = None
    for i in src:
        if i == '\n':
            newline += 1
        elif i in whitespaces_set:
            space = i
        else:
            if newline:
                buf.append('\n' * newline)
            elif space:
                buf.append(space)
            newline = 0
            space = None
            buf.append(i)
    if newline:
        buf.append('\n' * newline)
    elif space:
        buf.append(space)
    return ''.join(buf)


def zh_rate(src: str) -> float: 
    zh_char = re.compile(r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]')
    return len(re.findall(zh_char, src)) / len(src) if len(src) else 0

def eliminate_zh_space(src: str) -> str:
    """
    成句：
        对于中文，我们需要一个滑动窗口扫描每个字周围的字，
        由于双字词语最多，字越多的词语越少，我们需要一种函数来计算一个字和其他字的上下文相关度。
        我们仅删除字与字之间上下文相关度低的空格。或者这一步我们直接交给jieba

    """
    zh_no_concat_ruleset = [
        re.compile(r'摘要$'),
        re.compile(r'注$'),
        re.compile(r'导言$'),
        re.compile(r'^附件[一二三四五六七八九十].$'),
    ]
    zh_no_concat_ruleset_s2 = [
        re.compile(r'^[0-9]+\.'),
    ]
    def merge(buf: list, segment: list, use_jieba=True):
        def can_concat_two_by_ruleset(s1: str, s2: str) -> bool:
            if (r2 := zh_rate(s2)) <= 0.01 or (r1 := zh_rate(s1)) <= 0.01:
                return False

            if not use_jieba:
                return True

            back_char = s1[-1]
            front_char = s2[0]
            if back_char == '。': # 特判标点符号
                return False
            elif back_char in ('，', '）', '、'):
                return True
            
            match_no_concat_ruleset = False
            for pat in zh_no_concat_ruleset:
                if re.search(pat, s2) or re.search(pat, s1):
                    match_no_concat_ruleset = True
                    break
            if match_no_concat_ruleset:
                return False
            for pat in zh_no_concat_ruleset_s2:
                if re.search(pat, s2):
                    match_no_concat_ruleset = True
                    break
            if match_no_concat_ruleset:
                return False

            conn = back_char + front_char
            result = jieba.cut(s1 + s2, cut_all=True, HMM=False, use_paddle=True) # 开不开HMM实际上没有影响
            can_eliminate = False
            for r in result:
                if conn in r:
                    can_eliminate = True
                    break
            if can_eliminate:
                return True
            if r1 > 0.667 and r2 > 0.667:
                return True
            return False


        for i in segment:
            buf.append(i)
            while len(buf) >= 2 and can_concat_two_by_ruleset(buf[-2], buf[-1]):
                bck = buf.pop()
                buf[-1] += bck

    linebuf = []
    for line in src.split('\n'):
        seg = line.split(' ')
        segbuf = []
        merge(segbuf, seg, False)
        linebuf.append(' '.join(segbuf))

    linebuf2 = []
    merge(linebuf2, linebuf)

    return '\n'.join(linebuf2)

def cal_ppl_withBert_bs(model, tokenizer, text, device):
    with torch.no_grad():
        sentence_loss = torch.zeros(len(text))

        inputs = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(device)
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        text_length = (attention_mask.sum(dim=1) - 2).cpu()
        text_mask = attention_mask.clone().cpu()
        text_mask[:, 0] = 0
        text_mask.scatter_(1, text_length.unsqueeze(dim=1)+1, 0) #把cls和eos的mask都置为0
        max_length = attention_mask.sum(dim=-1).max()
        mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        #开头结尾是cls和eos，所以只改中间部分
        for i in range(1, max_length-1):
            mask_input = input_ids.clone()
            mask_input[:, i] = mask_id
            output = model(mask_input, attention_mask)
            prediction_scores = output[0]
            softmax = nn.Softmax(dim=1)
            ps = softmax(prediction_scores[:, i]).log()
            word_loss = torch.gather(ps, 1, input_ids[:, i].unsqueeze(dim=1)).cpu()
            sentence_loss += word_loss.squeeze() * text_mask[:, i] #超过长度的不参与计算loss

        ppl = np.exp(-sentence_loss/text_length)
        return ppl

def cal_ppl_withBert(model, tokenizer, text, device):
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(text)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sen_len = len(tokenize_input)
        sentence_loss = 0.

        for i, word in enumerate(tokenize_input):
            # add mask to i-th character of the sentence
            tokenize_input[i] = '[MASK]'
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)

            output = model(mask_input)

            prediction_scores = output[0]
            softmax = nn.Softmax(dim=0)
            ps = softmax(prediction_scores[0, i]).log()
            word_loss = ps[tensor_input[0, i]]
            sentence_loss += word_loss.item()

            tokenize_input[i] = word
        ppl = np.exp(-sentence_loss/sen_len)
        return ppl


def filter_high_ppl(lines, model, tokenizer, ppl_bs, ppl_thres, device):
    #将高于ppl_thes的文本行过滤掉
    low_ppl_lines = []
    st_idx = 0
    for st_idx in tqdm(range(0, len(lines), ppl_bs)):
        ed_idx = min(st_idx+ppl_bs, len(lines))
        temp_lines = lines[st_idx:ed_idx]
        scores = cal_ppl_withBert_bs(model, tokenizer, temp_lines, device)
        for i in range(len(temp_lines)):
            if len(temp_lines[i].split()) == 0 or scores[i] < ppl_thres:
                low_ppl_lines.append(temp_lines[i])
    return low_ppl_lines

def seg_lines(lines):
    #根据空行将文本分为若干段，每段以list形式保存每一行
    seg_lines = []
    n = len(lines)
    i = 0
    add_line = []
    while i < n:
        line = lines[i]
        if (i == n-1 or len(line.split())==0) and len(add_line) > 0:
            seg_lines.append(add_line)
            add_line = []
            i +=1 
        elif (i == n-1 or len(line.split())==0) and len(add_line) == 0:
            i += 1
        else:
            add_line.append(line)
            i += 1
    return seg_lines

def filter_split(seg_lines):
    #初步清理每行句子，合并换行符与空字符，合并多个连续空字符
    filtered_lines = []
    for paragraph in tqdm(seg_lines):
        new_paragraph = []
        for line in paragraph:
            dst = filter_duplicated_whitespaces(line)
            if len(dst) < 2:
                continue
            dst2 = eliminate_zh_space(dst)
            if dst2.endswith('\n'):
                dst2 = dst2[:-1]
            new_paragraph.append(dst2)
        filtered_lines.append(new_paragraph)
    return filtered_lines

def cat_lines_bySymbols(lines, model, tokenizer, cat_symbols, cat_bs, device):
    #用各种标点拼接同一段内的相邻行，包括，。！？等标点，以及代表分段的空格和直接连接
    #取拼接后ppl最小的结果作为最终结果
    #注意，如果一次考虑多行拼接，则计算量会指数增加，所以每次只考虑两行直接的连接
    cat_lines = []
    for paragraph in tqdm(lines):
        #只有一行的直接跳过
        if len(paragraph) == 1:
            cat_lines.append(paragraph[0])
            continue
        symbol_list = []    #记录各行间的连接标点
        for i in range(0, len(paragraph)-1, cat_bs):
            #先把cat_bs * len(cat_symbols)拼起来，构成候选项，然后选出ppl最小的一项
            line1_list = [paragraph[j] for j in range(i, min(i+cat_bs, len(paragraph)-1))]
            line2_list = [paragraph[j] for j in range(i+1, min(i+1+cat_bs, len(paragraph)))]
            line_list = [line1_list[j] + symbol + line2_list[j] for j in range(len(line1_list)) for symbol in cat_symbols]
            ppl_scores = np.array(cal_ppl_withBert_bs(model, tokenizer, line_list, device))
            ppl_scores = ppl_scores.reshape((-1, len(cat_symbols)))
            symbol_idx = np.argmin(ppl_scores, axis=1)
            for j in range(len(line1_list)):
                #如果前一个句子本身就以标点结尾，则直接沿用该标点，并把前一个句子的末尾去掉以防标点重复
                if line1_list[j][-1] in punctuation_set:
                    symbol_list.append(line1_list[j][-1])
                    paragraph[i+j] = paragraph[i+j][:-1]
                #否则使用ppl最小的标点
                else:
                    symbol = cat_symbols[symbol_idx[j]]
                    symbol_list.append(symbol)
        #利用选中的symbol，连接所有句子
        new_line = paragraph[0]
        for i in range(1, len(paragraph)):
            new_line = new_line + symbol_list[i-1] + paragraph[i]
        cat_lines.append(new_line)
    return cat_lines

def split_by_symbols(lines):
    #根据标点符号分句
    cleaned_lines = []
    for line in lines:
        split = re.split(r"([。！？])", line)
        if len(split) > 1:
            sentences = ["".join(i) for i in zip(split[0::2],split[1::2])]
        else:
            sentences = split
        cleaned_lines = cleaned_lines + sentences
    return cleaned_lines