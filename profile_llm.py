import torch
from transformers import AutoModelForCausalLM


# start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _ in range(3):
    # dummy input, batch size 16, sequence length 256
    inputs = torch.randint(0, 100, (16, 256), device="cuda") 
    loss = torch.mean(model(inputs).logits) # dummy loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# dump memory snapshot history to a pkl file and stop recording
torch.cuda.memory._dump_snapshot("profile_llm.pkl")
torch.cuda.memory._record_memory_history(enabled=None)
                                        