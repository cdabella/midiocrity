# midiocrity

## Populate the ./data folder
Run the following from the repo root to download the Clean MIDI subset of [The Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)

```
cd data
./getData.sh
```

## Environment Setup
To add the environment with conda, use the following commands:
```
conda create -n midiocrity python=3.9
conda activate midiocrity
pip install -r requirements.txt
```

## Training the model
To train the model, navigate to the midiocrity folder, and run:
```
python train.py
```

## Architecture
We are using a modified version of the architecture referenced in the [Off the Beaten Track](https://arxiv.org/pdf/1804.09808.pdf) paper.

![alt text](architecture.png "VAE interpolation Architecture")