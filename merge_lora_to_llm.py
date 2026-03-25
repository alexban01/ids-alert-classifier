from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER_PATH = "./v1-ids-lora-adapter"
MERGED_PATH = "./ids-model-merged"

print("Loading and merging...")
model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    device_map="cpu",        # merge on CPU to avoid VRAM issues
)
model = model.merge_and_unload()
model.save_pretrained(MERGED_PATH)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print(f"✅ Merged model saved to {MERGED_PATH}")
