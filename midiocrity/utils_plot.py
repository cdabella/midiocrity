from midiocrity_vae import MidiocrityVAE
from dataset import MidiDataloader
import json
import librosa.display

import yaml
from rich import print as rprint
import torch
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
import pretty_midi
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('../config/midiocrity_vae.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

with open('../data/metadata.json') as json_file:
    data = json.load(json_file)

def load_dummy_model():
    # rprint(config)

    # For reproducibility
    if 'seed' in config['train_params']:
        torch.manual_seed(config['train_params']['seed'])
        np.random.seed(config['train_params']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = MidiocrityVAE(
        # use_cuda = False,
        encoder_params={
            "n_cropped_notes": config['model_params']['n_cropped_notes'],
            "z_dim": config['model_params']['z_dim'],
            "phrase_size": config['model_params']['phrase_size'],
            "hidden_size": config['model_params']['encoder_params']['hidden_size'],
            "num_layers": config['model_params']['encoder_params']['num_layers'],
            "batch_size": config['data_params']['batch_size'],
            "n_tracks": config['model_params']['decoder_params']['n_tracks'],
        },
        decoder_params={
            "n_cropped_notes": config['model_params']['n_cropped_notes'],
            "z_dim": config['model_params']['z_dim'],
            "phrase_size": config['model_params']['phrase_size'],
            "hidden_size": config['model_params']['decoder_params']['hidden_size'],
            "num_layers": config['model_params']['decoder_params']['num_layers'],
            "batch_size": config['data_params']['batch_size'],
            "n_tracks": config['model_params']['decoder_params']['n_tracks'],
        }
    )
    return model
   

def load_model(PATH = '../models/MidiocrityVAE.epoch-19.pt'):
    model = load_dummy_model()    
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    return model

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def custom_postprocess( X_drums):
    X_drums = X_drums[:,:,:,0]
    #putting tracks back toghether
    batch_size = X_drums.shape[0]
    n_timesteps = X_drums.shape[1]

    # converting softmax outputs to categorical
    tracks = []
    for track in [X_drums]:
        track = to_categorical(track.argmax(2), num_classes=130)
        track = np.expand_dims(track, axis=-1)
        tracks.append(track)

    X = np.concatenate(tracks, axis=-1)

    # copying previous timestep if held note is on
    for sample in range(X.shape[0]):
        for ts in range(1, X.shape[1]):
            for track in range(X.shape[3]):
                if X[sample, ts, -1, track] == 1: # if held note is on
                    X[sample, ts, :, track] = X[sample, ts-1, :, track]

    X = X[:, :, :-2, 0]
    return X

def generate_loader():
    loader = MidiDataloader(
        config['data_params']['tensor_folder'],
        config['data_params']['batch_size'],
        config['data_params']['shuffle'],
        config['data_params']['num_workers'],
        batch_limit=config['data_params']['batch_limit']
    )
    loader.phase = 'test'
    return loader

def get_metadata():
    return data

# Based on https://avandekleut.github.io/vae/
def plot_latent_nolabel(autoencoder, data, num_batches=100):
    """Plot data over a latent space of an autoencoder

    Parameters:
    autoencoder (nn.Module): The autoencoder. Must have the encode function.
    data ((x, y)): 
        -- x(batch size, seq_len, input_size, ntrack)
        -- y(artist, song)
    num_batches: max number of batches

    Returns:
    None

    """
    plt.figure(figsize=(5,5))
    Zx =[]
    Zy =[]
    for i, x in enumerate(data):
        # Encode enpoints
        mu, lv = autoencoder.encode(x)
        z = autoencoder.reparameterize(mu, lv)
        z = z.to('cpu').detach().numpy()
        # plt.scatter(z[:, 0], z[:, 1], c='blue')
        Zx.append(z[0,0])
        Zy.append(z[0,1])
        if i > num_batches:
            print('hi')
            break
    plt.scatter(Zx, Zy)

    return (Zx, Zy)

def load2instrument(inst, track, phrase_len = 60, delay=0, shortener = 1):
    track_pp = custom_postprocess(track)
    track_pp = track_pp[:,0:int(256/shortener),:]
    phrase_len = phrase_len# //shortener
    delay = delay // shortener
    starts = np.linspace(0 + delay, phrase_len//shortener +delay, track_pp.shape[1])
    ends = starts + phrase_len / 255
    pitches = track_pp.argmax(axis=2)[0]
    mask = pitches!=0
    velocity = int(100)

    for pitch, start, end in zip(pitches[mask], starts[mask], ends[mask]):
        inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100, y_axis='cqt_note'):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis=y_axis,
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

def plot_encoded(data, num_batches=100):
    plt.figure(figsize=(5,5))
    for i, z in enumerate(data):
        plt.scatter(z[:, 0], z[:, 1], c='blue')
        if i > num_batches:
            plt.colorbar()
            break

def plot_latent(autoencoder, data, num_batches=100):
    """Plot data over a latent space of an autoencoder

    Parameters:
    autoencoder (nn.Module): The autoencoder. Must have the encode function.
    data ((x, y)): 
        -- x(batch size, seq_len, input_size, ntrack)
        -- y(artist, song)
    num_batches: max number of batches

    Returns:
    None

   """
    for i, (x, y) in enumerate(data):
        z = autoencoder.encode(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break