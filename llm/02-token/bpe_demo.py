import argparse
import sys, os
from collections import Counter

def get_pairs(tokens_list):
    """统计所有相邻 token 对的出现次数"""
    pairs = Counter()
    for tokens in tokens_list:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

def merge_pair(tokens_list, pair, new_token):
    """在所有序列中，把指定的 pair 合并成 new_token"""
    result = []
    for tokens in tokens_list:
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        result.append(merged)
    return result

def train_bpe(text, num_merges=10):
    """
    训练一个最简 BPE 分词器。

    参数：
        text: 训练文本
        num_merges: 合并几次（决定最终词表大小）

    返回：
        vocab: 最终词表
        merge_rules: 合并规则列表（按顺序）
    """
    # 把每个词拆成单个字符 + 词尾标记
    words = text.split()
    print(f"【words is】: {words}")
    tokens_list = [list(word) + ["</w>"] for word in words]
    print(f"tokens_list = {tokens_list}")

    # 初始词表
    vocab = set()
    for tokens in tokens_list:
        vocab.update(tokens)
    
    print(f"【训练文本】：\n {text}")
    print(f"【词的数量】：\n {len(words)}")
    print(f"初始词表 ({len(vocab)} 个): {sorted(vocab)}")
    print(f"\n开始 BPE 合并（共 {num_merges} 轮）：")
    print("=" * 65)

    merge_rules = []

    for step in range(num_merges):
        pairs = get_pairs(tokens_list)
        if not pairs:
            print(f"\n  轮次 {step + 1}: 没有更多的 pair 可以合并了，提前停止")
            break

        best_pair = pairs.most_common(1)[0]
        pair, count = best_pair
        new_token = pair[0] + pair[1]

        print(f"\n  轮次 {step + 1}: 合并 ('{pair[0]}', '{pair[1]}') → '{new_token}'  "
              f"（出现 {count} 次）")
        
        tokens_list = merge_pair(tokens_list, pair, new_token)
        vocab.add(new_token)
        merge_rules.append((pair, new_token))

        # 展示前几个词的当前分词状态
        preview = ["|".join(t) for t in tokens_list[:8]]
        if len(tokens_list) > 8:
            preview.append("...")
        print(f"  当前: {preview}")
    
    print("\n" + "=" * 65)
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    print(f"最终词表 ({len(vocab)} 个):")
    for i, v in enumerate(sorted_vocab):
        print(f"  {v}")
    # if len(sorted_vocab) > 20:
    #     print(f"  ... 以及 {len(sorted_vocab) - 20} 个更短的 token")
    
    return vocab, merge_rules

def tokenize_with_bpe(word, merge_rules):
    """用训练好的 BPE 规则对新词进行分词"""
    tokens = list(word) + ["</w>"]

    for pair, new_token in merge_rules:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                tokens = tokens[:i] + [new_token] + tokens[i + 2:]
            else:
                i += 1

    return tokens


def main():
    # 1. 动态获取当前 py 文件所在的绝对路径目录
    # __file__ 代表当前脚本文件的路径，os.path.dirname 拿到它所在的文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 拼接出同级目录下的 default_corpus.txt 绝对路径
    default_training_file_path = os.path.join(current_dir, "bpe_training_samples.txt")
    default_testing_file_path = os.path.join(current_dir, "bpe_testing_samples.txt")


    parser = argparse.ArgumentParser(description="从零手写 BPE 分词算法")
    parser.add_argument("--text", type=str, default=None, help="自定义训练文本")
    parser.add_argument(
        "--training_file", 
        type=str, 
        nargs="?",  # 表示这个参数后面可以跟着 0 个或 1 个值
        const=default_training_file_path,   # 当用户写了 --training_file 但后面没带值时，使用的常量值
        default=None,   # 当用户压根没写 --training_file 时，它的默认值
        help="自定义训练文本文件路径")
    parser.add_argument(
        "--testing_file", 
        type=str, 
        nargs="?",  # 表示这个参数后面可以跟着 0 个或 1 个值
        const=default_testing_file_path,   # 当用户写了 --testing_file 但后面没带值时，使用的常量值
        default=None,   # 当用户压根没写 --testing_file 时，它的默认值
        help="自定义测试文本文件路径")
    parser.add_argument("--merges", type=int, default=10, help="合并轮数（默认 10）")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.training_file:
        if not os.path.exists(args.training_file):
            print(f"错误: 找不到文件 '{args.training_file}'", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(args.training_file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"读取文件时发生错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = ("low low low low low "
                "lower lower "
                "newest newest newest newest newest newest "
                "widest widest widest")
    print(f"【成功加载训练文本，总字符数】: {len(text)}")
    print(f"【加载的训练文本内容】：\n{text}")
    
    vocab, rules = train_bpe(text, num_merges=args.merges)

    # ==================== 用训练好的分词器切新词 ====================
    # print("\n\n对新词进行分词（包括训练时未见过的词）：")
    # print("-" * 45)

    if args.testing_file:
        if not os.path.exists(args.testing_file):
            print(f"错误: 找不到文件 '{args.testing_file}'", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(args.testing_file, "r", encoding="utf-8") as f:
                testing_text = f.read()
        except Exception as e:
            print(f"读取文件时发生错误: {e}", file=sys.stderr)
            sys.exit(1)
    
    for sentence in testing_text.split():
        tokens = tokenize_with_bpe(sentence, rules)
        print(f"  '{sentence}'  → {tokens}")

    # for word in seen_words:
    #     tokens = tokenize_with_bpe(word, rules)
    #     print(f"  '{word}'  → {tokens}  （训练时见过）")

    # print()
    # for word in unseen_words:
    #     tokens = tokenize_with_bpe(word, rules)
    #     print(f"  '{word}'  → {tokens}  （训练时没见过！）")

if __name__ == "__main__":
    main()