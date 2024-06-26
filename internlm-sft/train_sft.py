from argparse import ArgumentParser

from peft.tuners.lora import LoraLayer
import copy
from transformers.utils import logging
import os
import torch
import transformers
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/model/qwen/Qwen1.5-0.5B-Chat")
    use_lora: Optional[bool] = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default="/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/general", metadata={
        "help": "Path to the training data."})
    source_length: int = field(default=256)
    target_length: int = field(default=256)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=True)
    num_train_epochs = 5


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}  # {'train': ['/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/general/rename.json']}
    extension = all_file_list[0].split(".")[-1]  # json/txt

    logger.info("load files %d number", len(all_file_list))
    # raw_datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
    raw_datasets = load_dataset(  # huggingface datasets  -->Union
        extension,  # "json"
        data_files=data_files,
        cache_dir=cache_dir,  # 构建数据集时会生成.arrow文件至cache_dir，是一种存储数据的二进制格式。这种文件格式通常用于更高效地存储和加载大型数据集，特别是在深度学习任务中。
                              # 这些 .arrow 文件包含了数据集中的样本，通常以表格的形式组织，每一行代表一个样本，每一列代表一个特征。.arrow 文件还可以包含数据的元数据信息，如数据类型、列名等。
    )['train']
    return raw_datasets


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str,
                       data_args: DataArguments) -> Dataset:
    """开多个进程对数据集的数据做处理，包括增加eos适配含有input和没有input的数据等"""
    logger.info("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logger.info("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        """对数据集中的每条数据使用该方法"""
        ins_data = examples['instruction']
        if 'input' not in examples.keys():
            input_data = [""] * len(ins_data)
        else:
            input_data = examples['input']
        output = examples['output']

        len_ = len(ins_data)

        # sources = []
        # targets = []

        # for i in range(len_):
        #     s_t = prompt_input.format_map({'instruction':ins_data[i],
        #                                    'input':input_data[i]}) if input_data[i] != "" else prompt_input.format_map({'instruction':ins_data[i]})
        #     sources.append(s_t)

        sources = [prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]}) if input_data[
                                                                                                        i] != "" else prompt_no_input.format_map(
            {'instruction': ins_data[i]})
                   for i in range(len_)]
        sources = [i[:data_args.source_length] for i in sources]  # i==item
        targets = [
            f"{example[:data_args.target_length - 1]}{tokenizer.eos_token}" for example in output]

        # sources = [prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #            for example in examples]
        # targets = [
        #     f"{example['output']}{tokenizer.eos_token}" for example in examples]

        input_output = preprocess(
            sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples

    # partial用于固定函数的一个或多个参数，并返回一个接受剩余参数的新函数。
    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)
    # dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
    dataset = dataset.map(
        function=generate_sources_targets_p,  #
        batched=True,  # 表示数据集将以批次的形式传递给 generate_sources_targets_p 函数。
        desc="Running tokenizer on train dataset",
        num_proc=20  # 并行处理的进程数量
    ).shuffle()
    return dataset


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments,
                             data_args: DataArguments) -> tuple:
    if training_args.use_deepspeed:

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True,

        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map='auto',
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True

        )

    if model_args.use_lora:
        logger.info("Loading model to Lora")

        from peft import LoraConfig, get_peft_model
        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "q_proj", "v_proj"
        ]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)  # 这样给model添加lora
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()

    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    return model, tokenizer


def train():
    # parser = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    output_dir = "/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/output"
    model_args, data_args, training_args = ModelArguments(), DataArguments(), TrainingArguments(output_dir)

    # 加载model和tokenizer, 配置是否使用deepspeed和是否使用lora(调用peft)
    model, tokenizer = load_model_and_tokenizer(
        model_args, training_args, data_args)

    # 使用上下文管理器确保指定的代码块在主进程中首先执行
    with training_args.main_process_first(desc="loading and tokenization"):
        # 开多个进程对数据集的数据做处理，包括增加eos适配含有input和没有input的数据等
        train_dataset = make_train_dataset(
            tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    # __call__用于将类的实例对象作为函数进行调用
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,  # collect_fn
                                           label_pad_token_id=IGNORE_INDEX
                                           )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,  # TrainingArguments
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)  # 用于从 train_dataset 或 eval_dataset 的元素列表中形成批次的函数,
    # 直接调transformers的DataCollator的
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    # logging.basicConfig(
    #     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    # )
    train()
