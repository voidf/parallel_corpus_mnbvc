import argparse
from functools import partial
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer
from datasets import load_dataset

from tqdm import tqdm
import re

from utils import cal_ppl_withBert, cal_ppl_withBert_bs, filter_duplicated_whitespaces, eliminate_zh_space, punctuation_set

def handle_line(model, row, idx, ppl_thres=400.0) -> list[str]:
    lines = row['zh'].split('\n')
    # lines = lines[:1000] #!这里只保留头1000行方便测试，使用时请注释掉

    #去除所有ppl过大的行，注意保留空行以方便分段
    non_symbol_lines = []
    for line in tqdm(lines):
        if len(line.split())==0:
            non_symbol_lines.append(line)
            continue
        else:
            score = cal_ppl_withBert(model, tokenizer, line, device)
            if score < ppl_thres:
                non_symbol_lines.append(line)


    #尝试处理非正常断行
    #按照空行将文本分成多个段落，每个段落的每一行用list存起来
    seg_lines = []
    n = len(non_symbol_lines)
    i = 0
    add_line = []
    while i < n:
        line = non_symbol_lines[i]
        if len(line.split())==0:
            i += 1
            continue
        elif i == n-1 or (len(non_symbol_lines[i+1].split()) == 0 and len(add_line)>0):
            add_line.append(line)
            seg_lines.append(add_line)
            add_line = []
            i += 1    
        else:
            add_line.append(line)
            i += 1

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

    #用各种标点拼接同一段内的相邻行，包括，。！？等标点，以及代表分段的空格和直接连接
    #取拼接后ppl最小的结果作为最终结果
    #注意，如果一次考虑多行拼接，则计算量会指数增加，所以每次只考虑两行直接的连接
    cat_symbols = ['，', '。', '、', '；', '！', '？', '']
    cat_lines = []
    for paragraph in tqdm(filtered_lines):
        if len(paragraph) == 1:
            cat_lines.append(paragraph)
            continue
        symbol_list = []
        for i in range(len(paragraph)-1):
            line1, line2 = paragraph[i], paragraph[i+1]
            if line1[-1] in punctuation_set:
                symbol_list.append(line1[-1])
                paragraph[i] = line1[:-1]
                continue
            line_list = [line1 + symbol + line2 for symbol in cat_symbols]            
            ppl_scores = cal_ppl_withBert_bs(model, tokenizer, line_list, device)
            symbol = cat_symbols[np.argmin(ppl_scores)]
            symbol_list.append(symbol)
        new_line = paragraph[0]
        for i in range(1, len(paragraph)):
            new_line = new_line + symbol_list[i-1] + paragraph[i]
        cat_lines.append(new_line)


    #根据标点符号分句
    cleaned_lines = []
    for line in cat_lines:
        split = re.split(r"([。！？])", line)
        if len(split) > 1:
            sentences = ["".join(i) for i in zip(split[0::2],split[1::2])]
        else:
            sentences = split
        cleaned_lines = cleaned_lines + sentences

    with open('zh_para.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines) + '\n' + '='*10 + '\n')

    # return cleaned_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="hfl/chinese-bert-wwm-ext")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()
    model.to(device)

    dataset = load_dataset("ranWang/UN_PDF_TEXT_DATA", split='randomTest')
    dataset.map(partial(handle_line, model), with_indices=True)
    # handle_line(model, )


    