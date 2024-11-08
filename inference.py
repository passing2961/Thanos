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
