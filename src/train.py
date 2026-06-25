import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer

from transformer import Transformer
from dataset import train_loader, val_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("t5-small")

VOCAB_SIZE = tokenizer.vocab_size

model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=2
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

EPOCHS = 3
model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch in train_loader:

        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        optimizer.zero_grad()

        decoder_input = tgt[:, :-1]

        output = model(
            src,
            decoder_input
        )

        target = tgt[:, 1:]

        loss = criterion(
            output.reshape(-1, VOCAB_SIZE),
            target.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}"
    )
    torch.save(model.state_dict(), "transformer.pth")

print("Model saved successfully!")