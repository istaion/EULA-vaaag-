import sys
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def main():
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
    else:
        input_text = sys.stdin.read().strip()
        if input_text.startswith("{"):
            import json
            input_text = json.loads(input_text).get("text", "")

    model_path = "model/mbart/mbart_model_0928"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "fr_XX"
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id["fr_XX"]
    model.eval()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=400,
            num_beams=4,
            forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
        )
    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print(output)

    del model, tokenizer, inputs
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
