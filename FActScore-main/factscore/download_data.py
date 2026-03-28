import argparse
import os
import subprocess
import torch
import tqdm
import transformers


def download_file(_id, dest, cache_dir):
    """
    从 Google Drive 或指定 URL 下载文件。
    :param _id: Google Drive 的文件 ID 或完整的 HTTPS URL。
    :param dest: 下载后的目标文件名。
    :param cache_dir: 缓存目录。
    """
    if os.path.exists(dest) or os.path.exists(os.path.join(cache_dir, dest)):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if os.path.exists(dest.replace(".zip", "")) or os.path.exists(os.path.join(cache_dir, dest.replace(".zip", ""))):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        # 直接通过 URL 下载
        command = """wget -O %s %s""" % (dest, _id)
    else:
        # 从 Google Drive 下载，包含确认码处理
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        # 如果是压缩包，下载后自动解压并删除原包
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))



def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """
    调整分词器和嵌入层的维度，并初始化新增 token 的权重。
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 使用现有 token 权重的平均值来初始化新 token
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def recover_instruct_llama(path_raw, output_path, device="cpu", test_recovered_model=False):
    """
    恢复 Instruct-LLaMA 模型。
    通过将原始 LLaMA 权重与权重差 (Weight Diff) 相加来恢复模型。
    """

    # 加载原始 LLaMA 模型
    model_raw = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # 加载权重差模型
    model_recovered = transformers.AutoModelForCausalLM.from_pretrained(
        "kalpeshk2011/instruct-llama-7b-wdiff",
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_raw = transformers.AutoTokenizer.from_pretrained(path_raw)
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    tokenizer_recovered = transformers.AutoTokenizer.from_pretrained("kalpeshk2011/instruct-llama-7b-wdiff")

    # 执行权重相加操作
    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if output_path is not None:
        # 保存恢复后的模型
        model_recovered.save_pretrained(output_path)
        tokenizer_recovered.save_pretrained(output_path)

    if test_recovered_model:
        # 可选：运行测试推理
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")

    return model_recovered, tokenizer_recovered


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--llama_7B_HF_path',
                        type=str,
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    download_file("1sbW6pkYl6cc9gooD4WLaeoFKcAj3poZu", "demos.zip", args.data_dir)
    download_file("155exEdKs7R21gZF4G-x54-XN3qswBcPo", "data.zip", args.data_dir)
    download_file("1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I", "enwiki-20230401.db", args.data_dir)

    if args.llama_7B_HF_path:
        recover_instruct_llama(args.llama_7B_HF_path, os.path.join(args.model_dir, "inst-llama-7B"))

    # download the roberta_stopwords.txt file
    subprocess.run(["wget https://raw.githubusercontent.com/shmsw25/FActScore/main/roberta_stopwords.txt"], shell=True)

    # move the files to the data directory
    subprocess.run(["mv demos %s" % args.data_dir], shell=True)
    subprocess.run(["mv enwiki-20230401.db %s" % args.data_dir], shell=True)

