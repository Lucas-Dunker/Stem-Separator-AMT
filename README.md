# Audio Source Separation For Automatic Music Transcription

## Overview
Source separation is the process of isolating individual
sounds in an auditory mixture of multiple sounds, and has
a variety of applications ranging from speech enhancement
and lyric transcription to digital audio production for
music. Furthermore, Automatic Music Transcription (AMT)
is the process of converting raw music audio into sheet music
that musicians can read. Historically, these tasks have
faced challenges such as significant audio noise, long training
times, and lack of free-use data due to copyright restric-
tions. However, recent developments in deep learning have
brought new promising approaches to building low-distortion
stems and generating sheet music from audio signals.
Using spectrogram masking, deep neural networks, and the
MuseScore API, we attempt to create an end-to-end pipeline
that allows for an initial music audio mixture (e.g...wav file)
to be separated into instrument stems, converted into MIDI
files, and transcribed into sheet music for each component
instrument.

Created as a final project for CS5100, Foundations of Artificial Intelligence, at Northeastern University.

## Getting Started
Our project can be split in three primary directories - `midi-to-note`, `stem-separation`, and `transcriber`. All directories have their own demo videos for showing their desired outputs, as well as any relevant examples or documentation for generating files from your own inputs.

### midi-to-note
This directory contains our approach for converting MIDI data into sheet music via a MuseScore CLI tool. Sheet music can be generated via altering the `input_file` and `output_file` variables in `midi_conversion_example.py`.

### stem-separation
This directory contains python files for creating, training, deploying, and evaluating a stem separation model using 7-second clips from the [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. Models can be trained from your own MUSDB18 samples, or you can use our pre-trained model for stem separation. 

Our stem separation model works for separating vocals **only**; if more accurate results or more stems are desired, you will likely want to train a model with the entirety of the MUSDB18 dataset rather than our subset.

An end-to-end pipeline for stem separation can be ran from `StemSeparationModel.py`. All relevant dependencies can be found in `environment.yml`. 

### transcriber 
This directory contains python files for creating, training, and deploying an Automatic Music Transcription (AMT) model. 

Our AMT model works best with piano, as that was the majority of its training data. For more accurate results with the above stem separation model, it is recommend to retrain the model on note-labelled vocal data. 

A pipeline for midi generation can be ran from `midi_generation.py`. All relevant dependencies can be found in `requirements.txt`.                                       

## Contributors
- [Lucas Dunker](https://github.com/Lucas-Dunker)
- [Akash Setti](https://github.com/asetti2002)
- [Bradford Derby](https://github.com/bderbs30)
- [Samarth Galchar](https://github.com/Samarthvg)
- [Shashank Jarmale](https://github.com/shashjar)

## Acknowledgements

Without an abundance of accessible online resources, none of our work would have been possible. 

  - [Source Separation Tutorial Landing Page](https://source-separation.github.io/tutorial/landing.html)
  - [ArXiv Paper on Source Separation](https://arxiv.org/pdf/1810.11520)
  - [ScienceDirect Article on Source Separation](https://www.sciencedirect.com/science/article/pii/S1877050920310152?via%3Dihub)
  - [IJERT Research on Audio Stems Separation](https://www.ijert.org/research/audio-stems-separation-using-deep-learning-IJERTV10IS030074.pdf)
  - [MUSDB Dataset for Source Separation](https://sigsep.github.io/datasets/musdb.html)
  - [Seetharaman Paper on 2D Fourier Transform](https://pseeth.github.io/public/papers/seetharaman_2dft_waspaa2017.pdf)
  - [Another ArXiv Paper on Source Separation](https://arxiv.org/pdf/1806.03185)
  - [NUSSL GitHub Repository](https://github.com/nussl/nussl)
  - [Tutorial on TF and Masking](https://source-separation.github.io/tutorial/basics/tf_and_masking.html)
  - [Cal Poly Thesis on Source Separation](https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=3064&context=theses)
  - [Stanford CS230 Project Report](https://cs230.stanford.edu/projects_spring_2020/reports/38948801.pdf)
  - [Wav2Mid GitHub Repository](https://github.com/jsleep/wav2mid)
  - [Final ArXiv Paper on Source Separation](https://arxiv.org/pdf/1710.11153)

TODO - Include References From Proposal
