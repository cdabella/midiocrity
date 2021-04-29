# import config
import config.config_preprocessing as config
import numpy as np
import time
import os
import pretty_midi as pm
import math
import random
import json
import pickle

# import progressbar
from rich.progress import track, Progress

import multiprocessing

from sklearn.model_selection import train_test_split

# Used to load HDF5 MDS files
# import tables

import matplotlib.pyplot as plt
import pypianoroll as pproll
import pprint
import itertools


# from keras.utils import to_categorical
import torch
from torch.utils.data import Dataset, DataLoader

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dataset")



def to_categorical(y, num_classes):
	""" 1-hot encodes a tensor """
	return np.eye(num_classes, dtype='uint8')[y]

pp = pprint.PrettyPrinter(indent=4)

path_sep = "/" if os.name == "posix" else "\\"


class MidiTorchDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


class MidiDataloader:
	def __init__(self, tensor_folder, batch_size, shuffle=True, num_workers=4,
				 train_valid_test_split=(0.7, 0.2, 0.1), seed=None, phase='train'):
		self.tensor_folder = tensor_folder
		self.tensor_files = [f for f in os.listdir(tensor_folder)]
		self.num_files = len(self.tensor_files)
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.num_workers = num_workers
		self.tvt_split = train_valid_test_split
		self.seed = seed

		if self.valid_phase(phase):
			self.phase = phase
		else:
			self.phase = 'train'

		if self.seed is not None:
			random.seed(self.seed)

		if self.shuffle:
			random.shuffle(self.tensor_files)

		self.train_files = self.tensor_files[:int(self.num_files * self.tvt_split[0])]
		self.valid_files = self.tensor_files[int(
			self.num_files * self.tvt_split[0]
		):int(
			self.num_files * (self.tvt_split[0] + self.tvt_split[1])
		)]
		self.test_files = self.tensor_files[self.num_files * (self.tvt_split[0] + self.tvt_split[1]):]

	def __iter__(self):
		files = getattr(self, f"{self.phase}_files")
		for file in files:
			filepath = os.path.join(self.tensor_folder, file)
			with open(filepath, 'rb') as f:
				X, y = pickle.load(f)
			dataset = MidiTorchDataset(X, y)
			dataloader = DataLoader(
				dataset,
				batch_size=self.batch_size,
				shuffle=self.shuffle,
				num_workers=self.num_workers
			)
			for batch in dataloader:
				yield batch

			del X, y, dataset, dataloader

	def set_phase(self, phase):
		if phase not in ['train', 'valid', 'test']:
			raise ValueError(f'Phase not in [test, validation, train]: {phase}')
		self.phase = phase

	@staticmethod
	def valid_phase(phase):
		return phase in ['train', 'valid', 'test']


class MidiDataset:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def select_batch(self, idx):
		X = np.load(os.path.join(self.dataset_path, "batches", "X", str(idx) + ".npy"))
		Y = np.load(os.path.join(self.dataset_path, "batches", "Y", str(idx) + ".npy"))
		label = np.load(os.path.join(self.dataset_path, "batches", "labels", str(idx) + ".npy"))
		return X, Y, label

	def select_song(self, idx, metadata=True):
		_metadata_ = None

		multitrack = pproll.load(os.path.join(self.dataset_path + "songs/" + str(idx) + ".npz"))

		if metadata:
			_metadata_ = self.retrieve_metadata(os.path.join(self.dataset_path, "metadata/", str(idx) + ".json"))

		return _metadata_, multitrack

	def select_pianoroll(self, idx):
		pianoroll = pproll.load(os.path.join(self.dataset_path + "pianorolls/" + str(idx) + ".npz"))
		return pianoroll

	def get_programs(self, song):	
		programs = np.array([ track.program for track in song.tracks ])
		return programs
	
	# I shape: (n_midi_programs, n_tracks)
	def programs_to_instrument_matrix(self, programs):    
		assert(len(programs) == self.n_tracks)
		
		I = np.zeros((self.n_midi_programs, self.n_tracks))
		for i, program in enumerate(programs):
			I[program, i] = 1

		return I

	# I shape: (n_midi_programs, n_tracks)
	def instrument_matrix_to_programs(self, I):
		assert(I.shape[1] == self.n_tracks)
		assert(I.shape[0] == self.n_midi_programs)

		programs = [ np.argmax(I[:, i]) for i in range(I.shape[1]) ]

		return np.array(programs)
	
	def retrieve_metadata(self, path):
		with open(path, "r") as fp:
			metadata = json.load(fp)
		return metadata

	def retrieve_pianoroll_metadata(self, meta_link, idx):
		if not isinstance(idx, str):
			raise TypeError("idx must be a string")
		song_id = meta_link[idx]
		return self.retrieve_metadata(os.path.join(self.dataset_path, "metadata/", str(song_id) + ".json"))

	def retrieve_instrument_matrix(self, path):	
		I = np.load(path)
		return I

	def generate_batches(self, path, filenames, batch_size):
		print("Generating batches from data...")
		dataset_len = len(filenames)
		# shuffle samples 
		random.shuffle(filenames)
		
		# discard filenames
		remainder = dataset_len % batch_size
		dataset = np.array(filenames[:-remainder])
		dataset_len = dataset.shape[0]

		assert(dataset_len % batch_size == 0)
		dataset = dataset.reshape((-1, batch_size))
		n_of_batches = dataset.shape[0]

		for i in range(n_of_batches):
			source = dataset[i, :]
			dest = []
			for sample in source:
				multitrack = pproll.load(os.path.join(path, sample))
				proll = multitrack.get_stacked_pianoroll()
				dest.append(proll)

			dest = np.array(dest)
			yield dest

	# warning: tends to use a lot of storage (disk) space
	def create_batches(self, batch_size=128):
		print("Building batches from data...")

		batch_path = os.path.join(self.data_path, "batches/")
		if not os.path.exists(batch_path):
			os.makedirs(os.path.join(batch_path, "X"))
			os.makedirs(os.path.join(batch_path, "Y"))
			os.makedirs(os.path.join(batch_path, "labels"))

		pianorolls_path = os.path.join(self.dataset_path, "pianorolls/")
		metadata_path = os.path.join(self.dataset_path, "metadata/")

		_, _, files = next(os.walk(pianorolls_path))
		
		dataset_len = len(files)

		random.shuffle(files)
		remainder = dataset_len % batch_size
		dataset = np.array(files[:-remainder])
		dataset_len = dataset.shape[0]

		print("dataset_length:", dataset_len)
		print("batch_size:", batch_size)
		print("number of batches:", dataset_len // batch_size)
		print("remainder:", remainder)

		assert(dataset_len % batch_size == 0)
		dataset = dataset.reshape((-1, batch_size))
		n_of_batches = dataset.shape[0]

		# store each batch in a file toghether
		# bar = progressbar.ProgressBar(max_value=n_of_batches)

		meta_link = json.load(open(os.path.join(self.dataset_path, "meta_link.json")))
		for i in track(range(n_of_batches), description="Batching data..."):
			# bar.update(i)
			source = dataset[i, :]
			dest = []
			labels = []
			# for each pianoroll, store it and the corresponding labels
			for sample in source:
				multitrack = pproll.load(os.path.join(pianorolls_path, sample))
				proll = multitrack.get_stacked_pianoroll()
				dest.append(proll)

				# retrieve corresponding s factors
				sample_id = sample.split(".")[0]
				song_id = meta_link[sample_id]
				label = np.load(os.path.join(self.dataset_path, "labels", str(song_id) + ".npy"))
				labels.append(label)

			dest = np.array(dest)
			labels = np.array(labels)
			# preprocess batch, get X and Y
			X, Y = self.preprocess(dest)
			# store everything
			np.save(os.path.join(batch_path, "X", str(i) + ".npy"), X)
			np.save(os.path.join(batch_path, "Y", str(i) + ".npy"), Y)
			np.save(os.path.join(batch_path, "labels", str(i) + ".npy"), labels)

	def pooled_tensor_batches(self, args):
		idx, source = args
		dest = []
		labels = []

		batch_path = os.path.join(self.data_path, "batches/")
		pianorolls_path = os.path.join(self.data_path, "pianorolls/")
		metadata_path = os.path.join(self.data_path, "metadata/")

		# for each pianoroll, store it and the corresponding labels
		for sample in source:
			multitrack = pproll.load(os.path.join(pianorolls_path, sample))
			proll = multitrack.get_stacked_pianoroll()
			dest.append(proll)

			# retrieve corresponding s factors
			sample_id = sample.split(".")[0]
			# song_id = meta_link[sample_id]
			# label = np.load(os.path.join(self.dataset_path, "labels", str(song_id) + ".npy"))
			# labels.append(label)

		dest = np.array(dest)
		labels = np.array(labels)
		# preprocess batch, get X and Y
		X, Y = self.preprocess(dest)
		# store everything
		with open(os.path.join(batch_path, f'tensors-{idx}.pkl'), 'wb') as f:
			pickle.dump((torch.from_numpy(X), torch.from_numpy(Y)), f, pickle.HIGHEST_PROTOCOL)

	def create_tensor_batches(self, batch_size=128, n_processes=10):
		print("Building batches from data...")

		batch_path = os.path.join(self.data_path, "batches/")
		if not os.path.exists(batch_path):
			os.makedirs(batch_path)

		pianorolls_path = os.path.join(self.data_path, "pianorolls/")
		metadata_path = os.path.join(self.data_path, "metadata/")

		# _, _, files = next(os.walk(pianorolls_path))
		files = [f for f in os.listdir(pianorolls_path)]

		dataset_len = len(files)

		random.shuffle(files)
		remainder = dataset_len % batch_size
		dataset = np.array(files[:-remainder])
		dataset_len = dataset.shape[0]

		print("dataset_length:", dataset_len)
		print("batch_size:", batch_size)
		print("number of batches:", dataset_len // batch_size)
		print("remainder:", remainder)

		assert (dataset_len % batch_size == 0)
		dataset = dataset.reshape((-1, batch_size))
		n_of_batches = dataset.shape[0]

		batches = [(idx, dataset[idx, :]) for idx in range(n_of_batches)]
		with Progress() as progress:
			task = progress.add_task("Batching data...", total=n_of_batches, visible=True)
			chunksize = 1
			with multiprocessing.Pool(processes=n_processes) as pool:
				for result in pool.imap(self.pooled_tensor_batches, batches, chunksize=chunksize):
					progress.advance(task, advance=chunksize)

	def preprocess(self, X):
		# if silent timestep (all 0), then set silent note to 1, else set
		# silent note to 0
		def pad_with(vector, pad_width, iaxis, kwargs):			
			# if no padding, skip directly
			if pad_width[0] == 0 and pad_width[1] == 0:
				return vector
			else:

				if all(vector[pad_width[0]:-pad_width[1]] == 0):
					
					pad_value = 1
				else:
					pad_value = 0

				vector[:pad_width[0]] = pad_value
				vector[-pad_width[1]:] = pad_value


		# adding silent note
		X = np.pad(X, ((0, 0), (0, 0), (0, 2), (0, 0)), mode=pad_with)

		# converting to categorical (keep only one note played at a time)
		tracks = []
		for t in range(self.n_tracks):
			X_t = X[:, :, :, t]
			X_t = to_categorical(X_t.argmax(2), num_classes=self.n_cropped_notes)
			X_t = np.expand_dims(X_t, axis=-1)

			tracks.append(X_t)
		
		X = np.concatenate(tracks, axis=-1)
		
		# adding held note
		for sample in range(X.shape[0]):
			for ts in range(1, X.shape[1]):
				for track in range(X.shape[3]):
					# check for equality, except for the hold note position (the last position)
					if np.array_equal(X[sample, ts, :-1, track], X[sample, ts-1, :-1, track]):
						X[sample, ts, -1, track] = 1

		#just zero the pianoroll where there is a held note
		for sample in range(X.shape[0]):
			for ts in range(1, X.shape[1]):
				for track in range(X.shape[3]):
					if X[sample, ts, -1, track] == 1:
						X[sample, ts, :-1, track] = 0

		# finally, use [0, 1] interval for ground truth Y and [-1, 1] interval for input/teacher forcing X
		Y = X.copy()
		X[X == 1] = 1
		X[X == 0] = -1

		return X, Y


	def postprocess(self, X_drums, X_bass, X_guitar, X_strings):
		#putting tracks back toghether
		batch_size = X_drums.shape[0]
		n_timesteps = X_drums.shape[1]
		
		# converting softmax outputs to categorical
		tracks = []
		for track in [X_drums, X_bass, X_guitar, X_strings]:
			track = to_categorical(track.argmax(2), num_classes=self.n_cropped_notes)
			track = np.expand_dims(track, axis=-1)
			tracks.append(track)
		
		X = np.concatenate(tracks, axis=-1)

		# copying previous timestep if held note is on
		for sample in range(X.shape[0]):
			for ts in range(1, X.shape[1]):
				for track in range(X.shape[3]):
					if X[sample, ts, -1, track] == 1: # if held note is on
						X[sample, ts, :, track] = X[sample, ts-1, :, track]

		X = X[:, :, :-2, :]
		return X

	def get_guitar_bass_drums(self, song):
		guitar_tracks = []
		bass_tracks   = []
		drums_tracks  = []
		string_tracks = []

		for i, track in enumerate(song.tracks):
			if track.is_drum:
				track.name="Drums"
				drums_tracks.append(i)
			elif track.program >= 0 and track.program <= 31:
				track.name="Guitar"
				guitar_tracks.append(i)
			elif track.program >= 32 and track.program <= 39:
				track.name="Bass"
				bass_tracks.append(i)
			else:
				string_tracks.append(i)

		return guitar_tracks, bass_tracks, drums_tracks, string_tracks

	def pooled_process_file(self, args):
		def check_four_fourth(time_sign):
			return time_sign.numerator == 4 and time_sign.denominator == 4

		idx, filepath = args
		fetch_meta = {}  # in this dict I will store the id of the corresponding metadata file

		store_meta = False
		pbc = 1
		yeah = 0
		max_bar_silence = 0

		processed_folder = os.path.join(self.data_path, "pianorolls/")
		path, file = os.path.split(filepath)

		artist = path.split(path_sep)[-1]
		filename = file.split(".")[0]

		# test 0: check keysignature = 4/4 always.
		try:
			pm_song = pm.PrettyMIDI(filepath)
		except Exception:
			# print(f'{idx} Not Pretty MIDI  {filepath}')
			return pbc, yeah, fetch_meta

		if not all([check_four_fourth(tmp) for tmp in pm_song.time_signature_changes]):
			return pbc, yeah, fetch_meta

		del pm_song  # don't need pretty midi object anymore, now i need pianorolls

		try:
			base_song = pproll.parse(filepath, beat_resolution=4)
		except Exception:
			return pbc, yeah, fetch_meta

		# find a guitar, a bass and a drum instrument
		guitar_tracks, bass_tracks, drums_tracks, string_tracks = self.get_guitar_bass_drums(base_song)

		try:
			assert (string_tracks)
		except AssertionError:
			return pbc, yeah, fetch_meta

		# if string_tracks:
		base_song.merge_tracks(string_tracks, mode="max", program=48, name="Strings",
							   remove_merged=True)

		# merging tracks change order of them, need to re-find the new index of Trio track
		guitar_tracks, bass_tracks, drums_tracks, string_tracks = self.get_guitar_bass_drums(base_song)

		# take all possible combination of guitar, bass and drums
		for guitar_track in guitar_tracks:
			for bass_track in bass_tracks:
				for drums_track in drums_tracks:
					# select only trio tracks (and strings)
					current_tracks = [drums_track, bass_track, guitar_track, -1]
					names = ["Drums", "Bass", "Guitar", "Strings"]

					# create temporary song with only that tracks
					song = pproll.Multitrack()
					song.remove_empty_tracks()

					for i, current_track in enumerate(current_tracks):
						song.append_track(
							pianoroll=base_song.tracks[current_track].pianoroll,
							program=base_song.tracks[current_track].program,
							is_drum=base_song.tracks[current_track].is_drum,
							name=names[i]
						)

					song.beat_resolution = base_song.beat_resolution
					song.tempo = base_song.tempo

					song.binarize()
					song.assign_constant(1)

					# Test 1: check whether a track is silent during all the song
					if song.get_empty_tracks():
						continue

					pianoroll = song.get_stacked_pianoroll()

					i = 0
					while i + self.phrase_size <= pianoroll.shape[0]:
						window = pianoroll[i:i + self.phrase_size, :, :]
						# print("window from", i, "to", i+self.phrase_size)

						# keep only the phrases that have at most one bar of consecutive silence
						# for each track
						bar_of_silences = np.array([0] * self.n_tracks)
						for track in range(self.n_tracks):
							j = 0
							while j + self.bar_size <= window.shape[0]:
								if window[j:j + self.bar_size, :, track].sum() == 0:
									bar_of_silences[track] += 1

								j += 1  # self.bar_size

						# if the phrase is good, let's store it
						if not any(bar_of_silences > max_bar_silence):
							# data augmentation, random transpose bar
							for shift in np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], 1,
														  replace=False):
								tmp = pproll.Multitrack()
								tmp.remove_empty_tracks()
								for track in range(self.n_tracks):
									tmp.append_track(
										pianoroll=window[:, :, track],
										program=song.tracks[track].program,
										name=config.instrument_names[song.tracks[track].program],
										is_drum=song.tracks[track].is_drum
									)

								tmp.beat_resolution = 4
								tmp.tempo = song.tempo
								tmp.name = str(yeah)
								tmp.name = f"{idx}_{yeah}"

								# breakpoint()
								tmp.transpose(shift)
								tmp.check_validity()
								# print(os.path.join(processed_folder, f"{idx}_{yeah}" + ".npz"))
								tmp.save(os.path.join(processed_folder, f"{idx}_{yeah}" + ".npz"))
								del tmp
								store_meta = True
								# adding link to corresponding metadata file
								fetch_meta[f"{idx}_{yeah}"] = idx
								yeah += 1

						i += self.bar_size
					del song

		del base_song
		return pbc, yeah, fetch_meta

	# preprocessing as in Hierarchical AE paper. (for lmd matched)
	def preprocess_dataset_clean(self, data_folder, early_exit=None, n_processes=5):
		# helper functions
		def check_four_fourth(time_sign):
			return time_sign.numerator == 4 and time_sign.denominator == 4

		processed_folder = os.path.join(self.data_path, "pianorolls/")
		if not os.path.exists(processed_folder):
			os.makedirs(processed_folder)


		indexed_files = []
		idx = 0
		for path, subdirs, files in os.walk(data_folder):
			for file in files:
				if file.split(".")[-1].lower() not in ["mid", "midi"]:
					continue
				indexed_files.append((
					idx,
					os.path.normpath(os.path.join(path, file))
				))
				idx += 1

		self.dataset_length = len(indexed_files)

		# assign unique id for each song of dataset
		print("Preprocessing songs...")
		pbc = 0
		yeah = 0
		fetch_meta = {}  # in this dict I will store the id of the corresponding metadata file
		max_bar_silence = 0
		with Progress() as progress:
			total = early_exit if early_exit else self.dataset_length
			indexed_files = indexed_files[:early_exit] if early_exit else indexed_files
			task = progress.add_task("Preprocessing songs...", total=total, visible=True)

			results = []
			chunksize = 1
			with multiprocessing.Pool(processes=n_processes) as pool:
				for result in pool.imap(self.pooled_process_file, indexed_files, chunksize=chunksize):
					result_pbc, result_yeah, result_fetch_meta = result
					pbc += result_pbc
					yeah += result_yeah
					fetch_meta.update(result_fetch_meta)
					progress.console.print(f"pbc: {pbc:10}  yeah: {yeah:10}")
					progress.advance(task, advance=chunksize)
