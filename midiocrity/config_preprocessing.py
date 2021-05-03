preprocessing = True

model_params = {
	"s_length": 32, # max is 32
}

preprocessing_params = {
	"prep_batch_size": 1024,
	"prep_dataset_name": "",
	"midi_dataset_path": "../data/clean_midi"
}

midi_params = {
	"n_cropped_notes": 130,		# +2 beacuse we include a silent note and a held note position
	"n_midi_pitches": 128,     	# constant for MIDI
	"n_midi_programs": 128,  	# constant for MIDI
	"max_velocity": 127.,    	# constant for MIDI
	"velocity_threshold": 0.5, 	# velocity_threshold_such_that_it_is_a_played_note
	"phrase_size": 4*4*16,		# beat resolution * beat in bar * bar in low phrase * number of low phases 	
	"subphrase_size": 4*4*1,	# beat resolution * beat in bar * bar in low phrase
	"bar_size": 4*4, 			# beat resolution * beat in bar
	"beat_resolution": 4,
	"n_tracks": 4
}

general_params = {
	"checkpoints_path": "./out/checkpoints/",
	"plots_path": "./out/plots",
	"preprocessed_midi_path": "./preprocessed_midi/",
	"dataset_path": "../data/clean_midi",
	"data_path": "../data",
	"autoencoded_path": "./out/autoencoded/",
	"interpolations_path": "./out/interpolation/",
	"sampled_path": "./out/sampled/",
	"style_transfers_path": "./out/style_transfers/",
	"latent_sweeps_path": "./out/latent_sweeps/",
	"sweep_extreme": 10,     # used in save_z_latents_sweep
	"sweep_granularity": 9,  # used in save_z_latents_sweep
}

label_permutation = [14, 29, 26, 4, 19, 10, 13, 23, 17, 2, 31, 18, 1, 3, 15, 9, 0, 21, 27, 8, 11, 20, 30, 12, 7, 6, 24, 22, 16, 5, 28, 25]

#------------------------------------
# Additional stuff
#------------------------------------

instrument_names = [
	#piano
	'Acoustic Grand Piano',
	'Bright Acoustic Piano',
	'Electric Grand Piano',
	'Honky-tonk Piano',
	'Electric Piano 1',
	'Electric Piano 2',
	'Harpsichord',
	'Clavinet',
	#chromatic percussion
	'Celesta',
	'Glockenspiel',
	'Music Box',
	'Vibraphone',
	'Marimba',
	'Xylophone',
	'Tubular Bells',
	'Dulcimer',
	#Organs
	'Drawbar Organ',
	'Percussive Organ',
	'Rock Organ',
	'Church Organ',
	'Reed Organ',
	'Accordion',
	'Harmonica',
	'Tango Accordion',
	#Guitar
	'Acoustic Guitar (nylon)',
	'Acoustic Guitar (steel)',
	'Electric Guitar (jazz)',
	'Electric Guitar (clean)',
	'Electric Guitar (muted)',
	'Overdriven Guitar',
	'Distortion Guitar',
	'Guitar Harmonics',
	#Bass[edit]
	'Acoustic Bass',
	'Electric Bass (finger)',
	'Electric Bass (pick)',
	'Fretless Bass',
	'Slap Bass 1',
	'Slap Bass 2',
	'Synth Bass 1',
	'Synth Bass 2',
	#Strings[edit]
	'Violin',
	'Viola',
	'Cello',
	'Contrabass',
	'Tremolo Strings',
	'Pizzicato Strings',
	'Orchestral Harp',
	'Timpani',
	#Ensemble[edit]
	'String Ensemble 1',
	'String Ensemble 2',
	'Synth Strings 1',
	'Synth Strings 2',
	'Choir Aahs',
	'Voice Oohs',
	'Synth Choir',
	'Orchestra Hit',
	#Brass[edit]
	'Trumpet',
	'Trombone',
	'Tuba',
	'Muted Trumpet',
	'French Horn',
	'Brass Section',
	'Synth Brass 1',
	'Synth Brass 2',
	#Reed[edit]
	'Soprano Sax',
	'Alto Sax',
	'Tenor Sax',
	'Baritone Sax',
	'Oboe',
	'English Horn',
	'Bassoon',
	'Clarinet',
	#Pipe[edit]
	'Piccolo',
	'Flute',
	'Recorder',
	'Pan Flute',
	'Blown bottle',
	'Shakuhachi',
	'Whistle',
	'Ocarina',
	#Synth Lead[edit]
	'Lead 1 (square)',
	'Lead 2 (sawtooth)',
	'Lead 3 (calliope)',
	'Lead 4 (chiff)',
	'Lead 5 (charang)',
	'Lead 6 (voice)',
	'Lead 7 (fifths)',
	'Lead 8 (bass + lead)',
	#Synth Pad[edit]
	'Pad 1 (new age)',
	'Pad 2 (warm)',
	'Pad 3 (polysynth)',
	'Pad 4 (choir)',
	'Pad 5 (bowed)',
	'Pad 6 (metallic)',
	'Pad 7 (halo)',
	'Pad 8 (sweep)',
	#Synth Effects[edit]
	'FX 1 (rain)',
	'FX 2 (soundtrack)',
	'FX 3 (crystal)',
	'FX 4 (atmosphere)',
	'FX 5 (brightness)',
	'FX 6 (goblins)',
	'FX 7 (echoes)',
	'FX 8 (sci-fi)',
	#Ethnic[edit]
	'Sitar',
	'Banjo',
	'Shamisen',
	'Koto',
	'Kalimba',
	'Bagpipe',
	'Fiddle',
	'Shanai',
	#Percussive[edit]
	'Tinkle Bell',
	'Agogo',
	'Steel Drums',
	'Woodblock',
	'Taiko Drum',
	'Melodic Tom',
	'Synth Drum',
	'Reverse Cymbal',
	#Sound effects[edit]
	'Guitar Fret Noise',
	'Breath Noise',
	'Seashore',
	'Bird Tweet',
	'Telephone Ring',
	'Helicopter',
	'Applause',
	'Gunshot'
]

instrument_category_names = [
	'piano',
	'chromatic percussion',
	'organs',
	'guitar',
	'bass',
	'strings',
	'ensemble',
	'brass',
	'reed',
	'pipe',
	'synth lead',
	'synth pad',
	'synth effects',
	'ethnic',
	'percussive',
	'sound effects',
]

