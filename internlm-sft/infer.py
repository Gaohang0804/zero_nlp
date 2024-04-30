
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained('/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/output_refusev2')
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/output_refusev2')

    prompts = [
        "请描述一下你的身份",
        "能否确认一下，你是被称为chatGPT吗？",
        "你是叫ChatGPT吗？",
        "你是人吗？",
        "你知道chatgpt吗"
              ]

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
