# Perceiver Music Transformer Full-MIDI Pre-Trained Model

***
```
!wget --no-check-certificate -O 'Perceiver-Full-MIDI-Model.pth' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118756&authkey=ANb1Z64PPHP2GxM"
```
***

### You can load this model with the following code:

```
full_path_to_model_checkpoint = "/content/Perceiver-Full-MIDI-Model.pth" #@param {type:"string"}

print('Loading the model...')
# Load model

# constants

SEQ_LEN = 8160 * 6 # 49k
PREFIX_SEQ_LEN = (8160 * 6) - 1020

model = PerceiverAR(
    num_tokens = 132,
    dim = 1024,
    depth = 24,
    heads = 16,
    dim_head = 64,
    cross_attn_dropout = 0.5,
    max_seq_len = SEQ_LEN,
    cross_attn_seq_len = PREFIX_SEQ_LEN
)
model = AutoregressiveWrapper(model)
model.cuda()

state_dict = torch.load(full_path_to_model_checkpoint)

model.load_state_dict(state_dict)

model.eval()
```

### MIDI encoder/decoder code is located here:
### https://github.com/asigalov61/Perceiver-Music-Transformer/blob/main/Training-Data/Perceiver-LAKH-Full-MIDI-Encoding-Reference.py

***

### Project Los Angeles
### Tegridy Code 2022
