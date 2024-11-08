# Thanos: Enhancing Conversational Agents with Skill-of-Mind-Infused Large Language Model

[🤖 Thanos-1B](https://huggingface.co/passing2961/Thanos-1B) | [🤖 Thanos-3B](https://huggingface.co/passing2961/Thanos-3B) | [🤖 Thanos-8B](https://huggingface.co/passing2961/Thanos-8B) | [📄 Arxiv](https://arxiv.org/abs/2411.04496) | [📕 PDF](https://arxiv.org/pdf/2411.04496)

> 🚨 Disclaimer: All models and dataset are intended to be used for research purposes only.

## 📰 News
- Thanos-1B|3B|8B has been released in 🤗 [Huggingface Models](https://huggingface.co/collections/passing2961/thanos-6711f7a74227c6088d5d88f8).
- Multifaceted Skill-of-Mind dataset has been released in 🤗 [Huggingface Datasets](https://huggingface.co/datasets/passing2961/multifaceted-skill-of-mind).
- The code for fine-tuning and evaluation will be released soon! Stay tuned!

## Multifaceted Skill-of-Mind Dataset 🧠🤹

**Multifaceted Skill-of-Mind** Dataset is the first publicly available *skill-of-mind*-annotated dialogue dataset, encompassing multi-turn, multifaceted conversational skills along with explanations across various interactive scenarios (e.g., long-term, counseling, task-oriented), all grounded in diverse social contexts (e.g., demographics, personas, rules of thumb). To build this dataset, we first collect 12 existing dialogue datasets that cover a wide range of social contexts and scenarios: [Soda](https://arxiv.org/abs/2212.10465), [Conversation Chronicles](https://arxiv.org/abs/2310.13420), [ProsocialDialogue](https://arxiv.org/abs/2205.12688), [EmpatheticDialogues](https://arxiv.org/abs/1811.00207), [Wizard of Wikipedia](https://arxiv.org/abs/1811.01241), [Cactus](https://arxiv.org/abs/2407.03103), [CaSiNo](https://arxiv.org/abs/2103.15721), [Multi-WOZ 2.2](https://aclanthology.org/2020.nlp4convai-1.13/), [PersuasionForGood](https://arxiv.org/abs/1906.06725), [Pearl](https://arxiv.org/abs/2403.04460), [Syn-PersonaChat](https://arxiv.org/abs/2312.10007), and [Stark](https://arxiv.org/abs/2407.03958). Next, we prompt GPT-4 (`gpt-4-turbo`) to annotate skill-of-mind on arbitrary turns within these dialogues. For the conversational skills, we develop a hierarchical taxonomy consisting of five main categories: (1) Interpersonal Skills, (2) Memory & Knowledge Management Skills, (3) Cognitive & Problem-Solving Skills, (4) Communication & Listening Skills, and (5) Task-Oriented Skills. For detailed descriptions of these categories, please refer to our paper.

```python
from datasets import load_dataset

ds = load_dataset("passing2961/multifaceted-skill-of-mind")
```

## Thanos: Skill-of-Mind-Infused LLM

### How to use

```python
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for prompt templates
THANOS_TEMPLATE = """### Task Description:
A dialogue and social context containing the speaker's demographics, preferences, persona, current situation/narrative, past dialogue summaries, episodic memory, or other relevant details are provided. During this dialogue, image-sharing moments may occur, represented by the format "[Sharing Image] <image_description>", where "<image_description>" represents the description of the shared image. Your task is to imagine yourself as the actual speaker who needs to respond in the next conversational turn. You will first generate the internal thought process behind selecting the appropriate conversational skill, and then generate the most appropriate conversational skill itself. The output format should be as follows: "### Explanation: (write an explanation for why the chosen skill is selected.) [RESULT SKILL] (A conversational skill that fits the situation.)"

### Social Context:
{social_context}

### Dialogue:
{dialogue}

### Explanation: """

THANOS_SYSTEM_MESSAGE = """You are an excellent skill predictor that generates the most appropriate conversational skill for the next turn response in the given dialogue and social context. Before generating the skill, please think about which skill is appropriate and then generate the skill."""

# Single Inference Example
def run_single_inference(model_path, social_context, dialogue):
    """
    Runs a single inference for conversational skill prediction based on social context and dialogue.
    
    Args:
        model_path (str): Path to the pre-trained model.
        social_context (str): Social context of the dialogue.
        dialogue (str): Dialogue text.
    
    Returns:
        tuple: A tuple containing the explanation and the predicted skill.
    """
    logging.info("Loading Thanos model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    ).eval()
    logging.info("Thanos model loaded successfully.")

    # Prepare messages for model input
    messages = [
        {"role": "system", "content": THANOS_SYSTEM_MESSAGE},
        {"role": "user", "content": THANOS_TEMPLATE.format(social_context=social_context, dialogue=dialogue)}
    ]

    # Tokenize the input
    logging.info("Tokenizing input for Thanos model...")
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to('cuda')

    # Generate output
    logging.info("Generating output from Thanos model...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            do_sample=True,
            temperature=0.9,
            max_new_tokens=256,
            top_p=1.0,
            use_cache=True,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and parse the output
    outputs = tokenizer.batch_decode(output_ids[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    logging.info("Output generated successfully.")

    if '[RESULT SKILL]' not in outputs:
        logging.warning("[RESULT SKILL] not found in output.")
        return "", ""

    explanation, skill = outputs.split('[RESULT SKILL]')
    return explanation.strip(), skill.strip()

if __name__ == "__main__":
    model_path = "passing2961/Thanos-8B"
    social_context = 'Tom and Sam are friends. They are talking about hamburgers.'
    dialogue = """Tom: What's your favorite burger?
    Sam: Definitely McDonald's. What about you?
    Tom: Hmm, I don't think so. I think Burger King is the best!"""

    explanation, skill = run_single_inference(model_path, social_context, dialogue)
    print("Explanation:", explanation)
    print("Skill:", skill)
    # Explanation: I'm curious about Tom's preference for Burger King since I prefer McDonald's. By asking him why he prefers Burger King, I can better understand his taste and preferences.
    # Skill: Clarification
```

## License and Recommendations

The **Multifaceted Skill-of-Mind** dataset is intended to be used for research purposes only. 

## Acknowledgement

This work was supported by a grant of the KAIST-KT joint research project through AI Tech Lab, Institute of convergence Technology, funded by KT [Project No. G01230605, Development of Task-oriented Persona-based Dialogue Generation Combining Multi-modal Interaction and Knowledge Modeling].

## Citation

If you find the resources in this repository useful, please cite our work:

```
@misc{lee2024thanosenhancingconversationalagents,
      title={Thanos: Enhancing Conversational Agents with Skill-of-Mind-Infused Large Language Model}, 
      author={Young-Jun Lee and Dokyong Lee and Junyoung Youn and Kyeongjin Oh and Ho-Jin Choi},
      year={2024},
      eprint={2411.04496},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.04496}, 
}
```
