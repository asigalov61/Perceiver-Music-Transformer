# Perceiver Music Transformer Training Data Packs

***

## Perceiver Multi-Instrumental Training Data Pack

### 1) Based on Euterpe Training Data
### 2) Uses Euterpe Mutli-Instrumental MIDI Encoding
### 3) Works with any type of data sampling and any model sequence length

```
!wget --no-check-certificate -O 'Euterpe-Training-Data.zip' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118738&authkey=AF6k171kUSX-Yrk"
```

### Encoding info:
### [(MIDI Channel(0-11) * 11)+Velocity(1-6), dTime(0-127)+128, Duration(1-127)+256, MIDI Pitch(1-127)+384]
### Compositions separator/Intro/Zero sequence: [0, 127+128, 127+256, 0+384]

***

## Perceiver Solo Piano Training Data Pack

### 1) Based on GIGA-Piano Training Data
### 2) Uses modified GIGA-Piano encoding
### 3) Works with any type of data sampling and any model sequence length

```
!wget --no-check-certificate -O 'Euterpe-Training-Data.zip' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118740&authkey=ANEK-9WanNFyalw"
```

### Encoding info:
### [dTime(0-126), Duration(1-126)+128, MIDI Pitch(1-126)+256, MIDI Velocity(1-126)+384]
### Compositions separator/Intro/Zero sequence: [126, 126+128, 0+256, 0+384]

***

### Project Los Angeles
### Tegridy Code 2022
