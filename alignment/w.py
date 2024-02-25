from collections import deque
import itertools
from datasets import load_dataset
import string
import pickle
import jieba
import re
import datetime

now_timer = datetime.datetime.now()


lang_list = ["ar", "en", "es", "fr", "ru", "zh"]
"""
一些发现：
- 拿到的文本连续一行的都是相关的
- 换段和回车符是相关的，有的回车符有换段意义，有的没有 （这个规则可能比较重要，将换行符和其他空格字符区别开来）
- 句号很重要，可以作为分段依据之一
- 其他标点符号也很重要，行尾的符号除句号外，可以作为删除换行的条件
- 连续出现的whitespace类型字符是没有意义的，所以可以换成一个
- 连续出现的相同标点符号可能是没有意义的，但是目录里的对齐符是为了保留格式
- 中文不同于其它5国语言，不会用空格隔开词语，所以需要一种方法删掉词语中多出来的空格
- 对于每一页，文章里都会出现模式相似的冗余噪声信息，比如页码，日期
- 这些文章数据里频繁出现有序列表，可以对这个特点写条件把那些长列表都提出来
- 英文中会出现i、ii、iii的罗马数字标号，且有可能有类似OCR扫描的错误，如iii打成Hi，需要一种计算编辑距离的机制

规则：
- 根据文件做前（后）向临近词频分析，对于每个单词，每次提取出仅包括本语言文字的连续的一行（见随后的正则），扫描紧随其后的单词统计这种临近词对个数，删回车时根据删除后临近两个词在全文中的出现频率是否大于某个阈值来决定
- 由于PDF排版性质，每行有最大字符限制，可以根据行的字数来决定这个换行要不要删（必要条件）
- 以逗号顿号反括号结尾的换行要删
- 以句号结尾的换行不删
- 以有序列表为开头的行都是新行，如其前面有换行则保留

工作方向：
1. 先做英语的段落。
2. 把其它语言的段落映射到英语去。


"""

regular_exp = {
    # \u0621-\u064A\u0660-\u0669
    # 除中文外，句子中都含空格
    'ar': re.compile(r'[\u0600-\u06ff ]+'),
    'zh': re.compile(r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]+'),
    'fr': re.compile(r'[a-zA-ZÀ-Ÿ ]+'),
    'es': re.compile(r'[a-zA-ZáéíóúñÁÉÍÓÚÑüÜ ]+'),
    'ru': re.compile(r'[А-я,Ё,ё ]+'),
    'en': re.compile(r'[A-Za-z ]+'),
}

period = {
    'zh': '。',
    'ar': '،',
    # 以下都是半角ascii句点
    'en': '.',
    'es': '.',
    'fr': '.',
    'ru': '.',
}

punctuations = {
    'ar': {
        '،': '.',  # full stop
        '.': '.',  # full stop
        '!': '!',  # exclamation mark
        '؟': '?',  # question mark
        '،': ',',  # comma
        '؛': ';',  # semicolon
        ':': ':',  # colon
        '“': '"',  # left quotation marks
        '”': '"',  # right quotation marks
    },
    'zh': {
        '，': ',',
        '。': '.',
        '：': ':',
        '？': '?',
        '！': '!',
        '；': ';',
        '“': '"',
        '”': '"',
        '（': '(',
        '）': ')',
    },
}

all_punctuation_set = set(string.punctuation)
for k, v in punctuations.items():
    all_punctuation_set.update(v.keys())

digits = {
    'ar': {
        '٠': 0,
        '١': 1,
        '٢': 2,
        '٣': 3,
        '٤': 4,
        '٥': 5,
        '٦': 6,
        '٧': 7,
        '٨': 8,
        '٩': 9,
    },
    'zh': {
        '零': 0,
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10,
    }
}


# symbol_set = set(string.punctuation + string.whitespace)
# def filter_duplicated_symbol(src: str) -> str:
#     buf = []
#     punc = []
#     space = []
#     for i in src:
#         if i in string.punctuation:
#             punc.append(i)
#         elif i in string.whitespace:
#             space.append(i)
#         else:
#             if punc:
#                 buf.append(punc[0])
#             elif space:
#                 buf.append(space[0])
#             punc.clear()
#             space.clear()
#             buf.append(i)
#     if punc:
#         buf.append(punc[0])
#     elif space:
#         buf.append(space[0])
#     return ''.join(buf)
whitespaces_set = set(string.whitespace.replace('\n', ''))


def filter_duplicated_whitespaces(src: str) -> str:
    """去噪：
        1. 如果换行符跟其它空格字符相连，这些字符替换成换行符
        2. 连续出现空格字符的，替换成其中一个空格字符"""
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

zh_no_concat_ruleset = [
    re.compile(r'摘要$'),
    re.compile(r'注$'),
    re.compile(r'导言$'),
    re.compile(r'^附件[一二三四五六七八九十].$'),
]
zh_no_concat_ruleset_s2 = [
    re.compile(r'^[0-9]+\.'),
]
zh_char = re.compile(r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]')
def zh_rate(src: str) -> float: return len(re.findall(zh_char, src)) / len(src) if len(src) else 0
def eliminate_zh_space(src: str) -> str:
    """
    成句：
        对于中文，我们需要一个滑动窗口扫描每个字周围的字，
        由于双字词语最多，字越多的词语越少，我们需要一种函数来计算一个字和其他字的上下文相关度。
        我们仅删除字与字之间上下文相关度低的空格。或者这一步我们直接交给jieba

    """
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

near_word = {}
context_length = 1 # 超参
score = [1] # 相关度赋分，保持长度与context_length一致
concat_thresold = 1
def eliminate_zh_breakline_prework(src: str) -> None:
    """统计字的上下文衔接度，可以分为用jieba分词后按词统计，也可以直接按字统计
    """
    return
    for line in src.split('\n'):
        # for cid, char in enumerate(line):
        #     if char in all_punctuation_set:
        #         continue
        #     char_stat = near_word.setdefault(char, {})
        #     for back_char_index in range(max(0, cid - context_length), cid):
        #         back_char = line[back_char_index]
        #         if back_char in all_punctuation_set:
        #             continue
        #         distance = cid - back_char_index
        #         char_stat[back_char] = char_stat.get(back_char, 0) + score[distance - 1]
        for zhseg in re.findall(regular_exp['zh'], line):
            sp = jieba.lcut(zhseg, use_paddle=True)
            for wid, word in enumerate(sp):
                word_stat = near_word.setdefault(word, {})
                for back_word_index in range(max(0, wid - context_length), wid):
                    back_word = sp[back_word_index]
                    dist = wid - back_word_index
                    word_stat[back_word] = word_stat.get(back_word, 0) + score[dist - 1]
                    
def eliminate_zh_breakline_mainwork(src: str) -> str:
    return src
    linebuf = []
    for line in src.split('\n'):
        if not linebuf or not re.search(regular_exp['zh'], line) or not re.search(regular_exp['zh'], linebuf[-1]):
            linebuf.append(line)
            continue
        s1 = linebuf[-1]
        s2 = line
        back_char = s1[-1]
        front_char = s2[0]
        # 不处理标点符号
        if back_char in all_punctuation_set or front_char in all_punctuation_set:
            linebuf.append(line)
            continue
        # 特判目录：阿拉伯数字和中文数字中的换行不处理
        if back_char in string.digits and front_char in digits['zh'] or \
            back_char in digits['zh'] and front_char in string.digits:
            linebuf.append(line)
            continue

        # 只看两个字接在一起
        # char_stat = near_word.setdefault(front_char, {}).get(back_char, 0)
        # if char_stat >= concat_thresold:
        #     linebuf[-1] += line
        # else:
        #     linebuf.append(line)

        back_word = jieba.lcut(s1, use_paddle=True)[-1]
        front_word = jieba.lcut(s2, use_paddle=True)[-1]
        word_stat = near_word.setdefault(front_word, {}).get(back_word, 0)
        if word_stat >= concat_thresold:
            linebuf[-1] += line
        else:
            linebuf.append(line)

    return '\n'.join(linebuf)

artifacts = [
    # 'attachgreatimportance',
    # 'r~pucicrllxcxl',
    # 'andtheoperative',
    # 'possibleissuestobecoveredbytheinstrument',
    # 'levelinindustrialized',
    # 'departures fromtheapproved calendar ofconferences thathave',
    # 'statement ofincomeandexpenditure forthebiennium',
    # 'thattheleveloftheoperational',
    # 'ofJudge',
    # 'Ihavethehonourtoenclose',
    # 'themisconduct',
    # 'objectspursued',
    # 'assessmentsapplicable',
    # 'since1974underUnitedNations',
    # 'wonderedwhether',
    # 'alsofacilitated',
    # '~eccmmends',
    # 'TheUnder-Secretary-Gene:al',
    # 'ofincomeandexpenditure',
    # 'ofthe',
]

dataset = load_dataset("ranWang/test_pdf_data", split='new')

filtered_file = []

indexes = []

EDIT_DISTANCE_THRESOLD = 3

def edit_distance(s1, s2):
    """chatgpt帮我写的n方编辑距离算法"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    if len(s2) - len(s1) > EDIT_DISTANCE_THRESOLD: # 优化相当大的O(1)剪枝
        return EDIT_DISTANCE_THRESOLD + 1
    
    # O(n)统计字符，进一步剪掉一些不必要用n^2编辑距离的情况 625s优化到22s
    char_distance = 0
    d = {}
    for s in s1:
        d[s] = d.get(s, 0) + 1
    for s in s2:
        d[s] = d.get(s, 0) - 1
    positive = 0
    negative = 0
    for k, v in d.items():
        if v > 0:
            positive += v
        else:
            negative += - v
    char_distance = max(positive, negative)
    if char_distance > EDIT_DISTANCE_THRESOLD:
        return char_distance

    distances = range(len(s1) + 1)

    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]

def read_int(s: str) -> int:
    "从开头开始读一段连续的数字"
    x = 0
    for c in s:
        if c.isdigit():
            x = x * 10 + int(c)
        else:
            return x
    return x

truncate_token = 'This record contains the text of speeches delivered in English and of the '
begin_token = re.compile(r'^The meeting was called to order at') # 151
begin_token2 = re.compile(r'^The meeting resumed at') # 6
suspend_token = re.compile(r'The meeting was suspended at')
rose_token = re.compile(r'The meeting rose at')


speaker_token = re.compile(r'^[A-Z].{2,25}( \(.*?\))?: ')

info_page_token = re.compile(r'United Nations .*?Corrections will be issued after the end of the session in a consolidated corrigendum\.', flags=re.M | re.S)
info_page_token2 = re.compile(r'United Nations .*?Corrected records will be reissued electronically on the Official Document System of the United Nations \(http://documents\.un\.org\)\.', flags=re.M | re.S)
info_page_token3 = re.compile(r'This record contains the text of speeches delivered in English.*?Corrected records will be reissued electronically on the Official Document System of the United Nations \(http://documents\.un\.org\)\.', flags=re.M | re.S)

def extract_sentences_from_single_file(filetext: list[str]):
    """把文件里的每个句子尽可能合并出来，输入是一个包含每页所有字符的列表，输出会把这个列表拍平
    输入应该每行事先做好strip"""

    """通过连续的标号来认段落，这个搞法比较激进"""
    match_infos = []
    line_marker = [] # 可以去掉换行的行数
    outputs = []
    lineno_pattern = re.compile(r'^[0-9]{1,3}\. [A-Z]')
    linedot_pattern = re.compile(r'^• [A-Z]')
    # for pageid, page in enumerate(filetext):
    flatten = list(itertools.chain(*[page.split('\n') for page in filetext]))
    for lineid, line in enumerate(flatten):
        m = re.match(lineno_pattern, line)
        if m:
            g = m.group(0)
            match_infos.append((read_int(g), lineid))
        m = re.match(linedot_pattern, line)
        if m:
            g = m.group(0)
            match_infos.append((-114514, lineid))
        
    for idx, (linecounter, lineid) in enumerate(match_infos[1:]):
        # 相邻两个识别头标号连续，则中间行的空格可以删掉（换成空格，将两段话拼在一起）
        prevcounter, previd = match_infos[idx]
        if linecounter == prevcounter + 1 or linecounter == prevcounter == -114514:
            line_marker.extend(range(previd, lineid - 1))

    line_marker.reverse()

    for lineid, line in enumerate(flatten):
        # if re.match(article_pattern, line) or outputs and re.match(article_pattern, outputs[-1]):
            # outputs.append(line)
            # continue
        while line_marker and line_marker[-1] < lineid - 1:
            line_marker.pop()

        if line_marker and lineid - 1 == line_marker[-1]:
            line_marker.pop()
            outputs[-1] += ' ' + line
        else:
            outputs.append(line)

    """根据观察，有至少三个因素影响一行结尾的回车能不能被删掉
    1. 次行首字母是不是小写字母
    2. 本行末尾字符是不是句号
    3. 本行是不是约有50个字符"""
    
    inputs: list[str] = outputs
    outputs = [inputs[0]]
    for lineid, nextline in enumerate(inputs[1:]):
        if not nextline:
            continue
        sc = 0 # 正表示删换行，负表示保留换行
        prevline = outputs[-1]
        if prevline[-1] == '.':
            sc -= 44
        if prevline[-1] == ',':
            sc += 81

        sc += min(60, len(inputs[lineid])) - 32

        if nextline[0].islower():
            sc += 83
        if re.match(lineno_pattern, nextline):
            sc -= 999
        if re.match(linedot_pattern, nextline):
            sc -= 999
        if re.match(speaker_token, nextline):
            sc -= 999


        if sc > 0:
            outputs[-1] += ' ' + nextline
        else:
            outputs.append(nextline)
    
    """将The meeting rose at ...后一直到The meeting was called to order...中间的部分去掉"""
    inputs: list[str] = outputs
    outputs = []
    accept_line = True
    for line in inputs:
        if accept_line:
            if re.search(rose_token, line):
                accept_line = False
            outputs.append(line)
        else:
            if re.match(begin_token, line) or re.match(begin_token2, line):
                accept_line = True
                outputs.append(line)

    output = '\n'.join(outputs)
    output = re.sub(info_page_token, '', output)
    output = re.sub(info_page_token2, '', output)
    output = re.sub(info_page_token3, '', output)
    # if 'This record contains the text of speeches delivered in English and of the interpretation of speeches delivered' in output:
        # print('bkp')
    return output


INDEX_TOKEN = '...'

maxed = 0

for rowid, row in enumerate(dataset):
    filtered_pages = {}
    # for lang in lang_list:
    for lang in ['en']:
        lang_match_file_content = row["content"][lang]
        bad = 0
        # if lang == 'en' or lang == 'zh':
        for page in lang_match_file_content:
            dst = page
            for ar in artifacts:
                if ar in dst:
                    bad = 1
                    err = ar
                    break
            if bad: break
                # with open('')
        file_index_titles = []
        current_index_title_id = 0
        def filter_index_title(page: str):
            """把正文里的目录索引条目拿掉"""

            global current_index_title_id, maxed
            filtered_page = []
            unmatched = deque()
            for lineid, line in enumerate(pageline := page.split('\n')):
                line = line.strip()
                matched = False
                # if current_index_title_id < len(file_index_titles):
                    # file_index_title = file_index_titles[current_index_title_id]
                    # cid = current_index_title_id
                for cid, file_index_title in enumerate(file_index_titles): # 每个都for太慢了，几十秒一个pdf
                    if (ed := edit_distance(file_index_title, line)) <= EDIT_DISTANCE_THRESOLD:
                        while unmatched: filtered_page.append(unmatched.popleft())
                        matched = True
                        print(file_index_title, ed, 'cid:', cid)
                        break
                    else:
                        if unmatched:
                            back_line = unmatched.pop()
                            # 如果要改三行的话这里要修改一下
                            if (ed := edit_distance(back_line + ' ' + line, file_index_title)) <= EDIT_DISTANCE_THRESOLD:
                                while unmatched: filtered_page.append(unmatched.popleft())
                                matched = True
                                print(file_index_title, ed, 'cid:', cid)
                                break
                            else:
                                unmatched.append(back_line)
                if matched:
                    current_index_title_id += 1
                    maxed = max(maxed, ed)
                else:
                # if not matched:
                    unmatched.append(line)
                    while len(unmatched) > 1:
                        filtered_page.append(unmatched.popleft())
            while unmatched: filtered_page.append(unmatched.popleft())
            return '\n'.join(filtered_page)

        if not bad:
            # 第一次过滤


            for pageid, page in enumerate(lang_match_file_content):
                lines = []

                # if page.count('.......')
                dot_count = 0
                pageline = page.split('\n')
                # 预先过滤：去掉满足begin_token之前的文件首部不成句信息
                # if pageid == 0:
                #     for truncated_title_pageline_index, line in enumerate(pageline):
                #         if re.match(begin_token, line) or re.match(begin_token2, line):
                #             break
                #     pageline = pageline[truncated_title_pageline_index:]


                
                for lineid, line in enumerate(pageline):
                    line = line.strip()
                    # if lineid < 4 or len(pageline) - lineid < 3: # discard pagination info
                    #     line = re.sub(r'([a-zA-Z0-9\.]{1,13}/){2,5}[A-Za-z0-9\.]{1,13}', '', line)
                    #     line = re.sub(r'^([0-9/]{1,8} ){0,1}[0-9-]{1,8}( [/0-9]{1,8}){0,1}$', '', line)
                    #     line = line.strip()
                    #     line = re.sub(r'^(\([PartVol\.]{1,4} [IVX]{1,4}\)\*?)$', '', line)
                    #     line = re.sub(r'^Article [IVX]{1,4}$', '', line) # 拿掉Article索引

                    # line = line.strip()
                    # line = re.sub(r'^\*[0-9]+\*$', '', line)
                    # line = re.sub(r'^[0-9]+-[0-9]+ \(E\)$', '', line)
                    if line:
                        lines.append(line)
                    if INDEX_TOKEN in line:
                        dot_count += 1
                
                if dot_count >= 4: # 有大于4行三个点的我们认为是目录页，用特别的处理方式或者先跳过
                    # indexes.append(dst)
                    unmatched = []

                    for line in lines:
                        line = line.strip().replace('\ufffe', '-') # 瞪眼法得，\ufffe应该是连词符-
                        if INDEX_TOKEN in line:
                            title: str = line.split(INDEX_TOKEN, 1)[0].strip()
                            done = 0

                            # 有个特征是标题总是有一个带.的标号
                            for rid in range(len(unmatched), max(len(unmatched) - 4, -1), -1):
                                concat_title = ' '.join([*unmatched[rid:], title])
                                dot_pos = concat_title.find('.')
                                if dot_pos != -1 and dot_pos < 6:
                                # if '.' in title:
                                    file_index_titles.append(concat_title)
                                    done = 1
                                    break # 没找到就取title
                            if not done:
                                file_index_titles.append(title)
                            unmatched.clear()
                        else:
                            unmatched.append(line)
                    # lang_match_file_content[pageid] = '' # 拿掉目录页
                else:

                    dst = '\n'.join(lines)
                    lang_match_file_content[pageid] = dst

            for pageid, page in enumerate(lang_match_file_content):
                dst = page
                # dst = filter_index_title(page)
                if dst:
                    filtered_pages.setdefault(lang, []).append(dst)
                    # pass
                    # continue

                # dst = page
        #     dst = filter_duplicated_whitespaces(page)
        #     if len(dst) < 2:
        #         continue

        #     if lang == 'zh':
        #         dst = eliminate_zh_space(dst)
        #         eliminate_zh_breakline_prework(dst)
        with open(f"pdf/{lang}{rowid}.pdf", 'wb') as pdf:
            pdf.write(row["meta"][lang])
        # if bad:
        #     with open(f"pdf/bad/{lang}{rowid}.pdf", 'wb') as pdf:
        #         pdf.write(row["meta"][lang])
        #     with open(f"pdf/bad/{lang}{rowid}.txt", 'w', encoding='utf-8') as txt:
        #         txt.write('=========='.join(lang_match_file_content))
        #         txt.write('\n\nBad reason: ' + err)
        #         print(rowid, err)
    filtered_file.append(filtered_pages)
    # print(rowid, (datetime.datetime.now() - now_timer).total_seconds())

# with open('r1.pkl', 'wb') as f:
#     pickle.dump(filtered_dataset, f)

# with open(f'near_word{context_length}.pkl', 'wb') as f:
#     pickle.dump(near_word, f)

# ================ 

# with open(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\r1.pkl', 'rb') as f:
#     filtered_dataset = pickle.load(f)
# with open(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\near_word' + f'{context_length}.pkl', 'rb') as f:
#     near_word = pickle.load(f)

dd = {}

# with open('indexes.txt', 'w', encoding='utf-8') as f:
#     f.write('++++++++++++++++++++++\n'.join(indexes))

for fi in filtered_file:
    for l, content in fi.items():
        # for p, i in enumerate(content):
            # content[p] = eliminate_zh_breakline_mainwork(i)
        dd.setdefault(l, []).append('=========='.join(content))  # 页
        # dd.setdefault(l, []).append(extract_sentences_from_single_file(content))  # 页

for l, content in dd.items():
    with open(f'{l}.txt', 'w', encoding='utf-8') as f:
        for fileid, filecontent in enumerate(content):
            f.write(f'<<<<<<<<<<{fileid}\n' + filecontent)
        # f.write('\n<<<<<<<<<<\n'.join(content))  # 文件

# print(filtered_dataset)

print('running time:', (datetime.datetime.now() - now_timer).total_seconds())
print('maxed:', maxed)

"""
I want you to align two paragraph sentence by sentence (chinese english). The chinese corpus may includes some noise, in this case, you should filter it out and follow the english sentence. Here is an example:

我是法国人
I am French

Are you ready for this task?
"""