# ViT (Vision Transformer) Implementation With PyTorch

<div align="center">
    <a href="">
        <img alt="open-source-image"
        src="https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F_Open_Source-%2350C878?style=for-the-badge"/>
    </a>
    <a href="https://www.youtube.com/watch?v=Vonyoz6Yt9c&t=2s">
        <img alt="youtube-tutorial"
        src="https://img.shields.io/badge/YouTube_Tutorial-grey?style=for-the-badge&logo=YouTube&logoColor=%23FF0000"/>
    </a>
</div>
<br/>
<div align="center">
    <p>Liked our work? give us a ⭐!</p>
</div>

<p align="center">
  <img src="./assets/arc.png" height="70%" width="70%"/>
</p>

This repository contains unofficial implementation of ViT (Vision Transformer) that is introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) using PyTorch. Implementation has tested using the [MNIST Dataset](https://www.kaggle.com/competitions/digit-recognizer) for image classification task.

## YouTube Tutorial
This repository also contains a corresponding YouTube tutorial with the title:
<br>
<br>
**Implement and Train ViT From Scratch for Image Recognition - PyTorch**

<div align="center">
    <a href="https://www.youtube.com/watch?v=Vonyoz6Yt9c&t=2s">Implement and Train ViT From Scratch for Image Recognition - PyTorch</a>
    <a href="https://www.youtube.com/watch?v=Vonyoz6Yt9c&t=2s">
        <img src="./assets/notebook-thumbnail.png" height="85%" width="85%%"/>
    </a>
</div>

## Table of Contents
* [ViT Implementation](#vitimp)
    * [ViT](#vit)
    * [PatchEmbedding](#embed)
* [Train Loop](#trainloop)
* [Usage](#usage)
* [Contact](#contact)

## ViT Implementation <a class="anchor" id="vitimp"></a>
We need two classes to implement ViT. First is the `PatchEmbedding` to processing the image and embeddings until we feed the transformer encoder Second is the `ViT` for the rest of the process. 


### ViT <a class="anchor" id="vit">

```
class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x
```

### PatchEmbedding <a class="anchor" id="embed">

```
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2))

        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x
```

## Train Loop <a class="anchor" id="trainloop"></a>
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

start = timeit.default_timer()
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_label["image"].float().to(device)
        label = img_label["label"].type(torch.uint8).to(device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = criterion(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)
            val_running_loss += loss.item()
    val_loss = val_running_loss / (idx + 1)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
    print(f"Valid Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print("-"*30)

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")
```

## Usage <a class="anchor" id="usage"></a>

You can run the code by downloading the notebook and updating the paths of `train` and `test` datasets.

You can also view it directly from https://www.kaggle.com/code/uygarkk/youtube-vit-implementation

## Contact <a class="anchor" id="contact"></a>
You can contact me with this email address: uygarsci@gmail.com
