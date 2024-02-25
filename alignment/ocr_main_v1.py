import argparse
import torch
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
import re

from utils import cal_ppl_withBert, cal_ppl_withBert_bs, filter_duplicated_whitespaces, eliminate_zh_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument('--input_path', type=str, default="/home/yuchengze/OCR/zh/zh.txt")
    parser.add_argument('--output_path', type=str, default="/home/yuchengze/OCR/zh/zh_head1000_cleaned_v1.txt")
    parser.add_argument('--ppl_thres', type=float, default=400.0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()
    model.to(device)

    #输入文本
    with open(args.input_path, 'r')  as f:
        lines = f.readlines()

    lines = lines[:1000] #!这里只保留头1000行方便测试，使用时请注释掉

    #去除所有ppl过大的行，注意保留空行以方便分段
    non_symbol_lines = []
    for line in tqdm(lines):
        if len(line.split())==0:
            non_symbol_lines.append(line)
            continue
        else:
            score = cal_ppl_withBert(model, tokenizer, line, device)
            if score < args.ppl_thres:
                non_symbol_lines.append(line)


    #根据空行将文本分段，每段内部直接拼接
    seg_lines = []
    n = len(non_symbol_lines)
    i = 0
    add_line = ''
    while i < n:
        line = non_symbol_lines[i]
        if line == "\n":
            i += 1
            continue
        elif i == n-1 or (len(non_symbol_lines[i+1].split()) == 0 and len(add_line)>0):
            add_line += line
            add_line = ' '.join(add_line.split('\n'))
            seg_lines.append(add_line)
            add_line = ''
            i += 1    
        else:
            add_line += line
            i += 1

    #清洗每段文本，去除多余的空格和\n
    filtered_lines = []
    for line in tqdm(seg_lines):
        lang = 'zh'
        dst = filter_duplicated_whitespaces(line)
        if len(dst) < 2:
            continue
        if lang == "zh":
            dst2 = eliminate_zh_space(dst)
        filtered_lines.append(dst2)

    #根据标点符号分句
    cleaned_lines = []
    for line in filtered_lines:
        split = re.split(r"([。！？])", line)
        if len(split) > 1:
            sentences = ["".join(i) for i in zip(split[0::2],split[1::2])]
        else:
            sentences = split
        cleaned_lines = cleaned_lines + sentences

    #输出结果
    with open(args.output_path, 'w') as f:
        for line in cleaned_lines:
            f.write(line+'\n')