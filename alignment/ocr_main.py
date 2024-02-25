import argparse
import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils import filter_high_ppl, seg_lines, filter_split, cat_lines_bySymbols, split_by_symbols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument('--input_path', type=str, default="./OCR/zh/zh.txt")
    parser.add_argument('--output_path', type=str, default="./OCR/zh/zh_head1000_cleaned.txt")
    parser.add_argument('--ppl_thres', type=float, default=100.0)
    parser.add_argument('--ppl_bs', type=int, default=16, help="通过ppl剔除乱码时的batch size")
    parser.add_argument('--cat_bs', type=int, default=4, help="通过ppl判断用哪种标点连接时的batch size，注意每次输入的数据量为(cat_bs * 候选标点数目)")

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
    low_ppl_lines = filter_high_ppl(lines, model, tokenizer, args.ppl_bs, args.ppl_thres, device)

    #尝试处理非正常断行
    #按照空行将文本分成多个段落，每个段落的每一行用list存起来
    segmented_lines = seg_lines(low_ppl_lines)

    #初步清理每行句子，合并换行符与空字符，合并多个连续空字符
    filtered_lines = filter_split(segmented_lines)

    #用各种标点拼接同一段内的相邻行，包括，。！？等标点，以及代表分段的空格和直接连接
    #取拼接后ppl最小的结果作为最终结果
    #注意，如果一次考虑多行拼接，则计算量会指数增加，所以每次只考虑两行直接的连接
    cat_symbols = ['，', '。', '、', '；', '！', '？', '']
    cat_lines = cat_lines_bySymbols(filtered_lines, model, tokenizer, cat_symbols, args.cat_bs, device)

    #根据标点符号分句
    cleaned_lines = split_by_symbols(cat_lines)

    #保存结果
    with open(args.output_path, 'w') as f:
        for line in cleaned_lines:
            f.write(line+'\n')