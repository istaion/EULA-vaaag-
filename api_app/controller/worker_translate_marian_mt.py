import sys
import torch
from transformers import MarianMTModel, MarianTokenizer

def main():
    # Lire le texte à traduire (via argument ou stdin)
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
    else:
        input_text = sys.stdin.read().strip()
        if input_text.startswith("{"):
            import json
            input_text = json.loads(input_text).get("text", "")

    model_path = "model/marianmt-vieux-francais-model2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement modèle/tokenizer
    model = MarianMTModel.from_pretrained(model_path).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model.eval()

    # Inférence
    batch = tokenizer.prepare_seq2seq_batch([input_text], return_tensors="pt", max_length=128, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        gen = model.generate(**batch, max_length=400, num_beams=4)
    output = tokenizer.decode(gen[0], skip_special_tokens=True)

    print(output)

    # Libération mémoire
    del model, tokenizer, batch
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
