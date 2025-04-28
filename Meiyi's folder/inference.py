import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login

def generate_text_from_image(image_path, prompt_text="Describe this image in detail"):
    # Load model and processor
    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Prepare multimodal input
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=text,
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generation_params = {
        "max_new_tokens": 64,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 1.0,
        "eos_token_id": stop_token_ids,
        "disable_compile": True
    }
    
    outputs = model.generate(**inputs, **generation_params)
    outputs_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    decoded_text = processor.batch_decode(outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return decoded_text

# Usage
if __name__ == '__main__':
    HF_TOKEN = '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'
    print('Loginning in HF')
    login(HF_TOKEN)
    print('Done')
    image_path = "../data/food_images/zucchini-noodles-with-anchovy-butter-56389802.jpg"
    result = generate_text_from_image(image_path)
    print("Generated Description:\n", result)
