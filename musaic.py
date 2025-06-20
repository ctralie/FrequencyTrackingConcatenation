"""
Code copyright Christopher J. Tralie, 2024
Attribution-NonCommercial-ShareAlike 4.0 International


Share — copy and redistribute the material in any medium or format
The licensor cannot revoke these freedoms as long as you follow the license terms.

 Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    NonCommercial — You may not use the material for commercial purposes .
    NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for my new musaicing technique
"""
import numpy as np
import argparse
import sys
sys.path.append("src")
from audioutils import load_corpus
from particle import ParticleAudioProcessor
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Path to audio file or directory for source sounds")
    parser.add_argument("--target", type=str, required=True, help="Path to audio file for target sound")
    parser.add_argument("--hopLength", type=int, default=512, help="Window Size in samples")
    parser.add_argument("--binsPerOctave", type=int, default=12, help="Number of CQT bins per octave")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument("--minFreq", type=int, default=32.7, help="Minimum frequency to use (in hz)")
    parser.add_argument("--maxFreq", type=int, default=8000, help="Maximum frequency to use (in hz)")
    parser.add_argument("--maxShift", type=int, default=2, help="Maximum CQT bins to shift up or down")
    parser.add_argument("--stereo", type=int, default=1, help="0-Mono, 1-Stereo Particle Left Only (Default), 2-Stereo Particles Each Channel")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use, or \"np\" for numpy")
    parser.add_argument("--nThreads", type=int, default=0, help="Use this number of threads in torch if specified")
    parser.add_argument("--r", type=int, default=7, help="Width of the repeated activation filter")
    parser.add_argument("--p", type=int, default=10, help="Number of simultaneous activations to use in particle filter")
    parser.add_argument("--pFinal", type=int, default=0, help="Number of simultaneous activations to use in output (by default use all of them)")
    parser.add_argument("--pd", type=float, default=0.95, help="Probability of sticking to an activation (0 is no stick, closer to 1 is longer continuous activations)")
    parser.add_argument("--L", type=int, default=10, help="Number of KL iterations")
    parser.add_argument("--alpha", type=float, default=0.1, help="L2 penalty for shrinking quiet activations")
    parser.add_argument("--particles", type=int, default=2000, help="Number of particles in the particle filter")
    parser.add_argument("--useTopParticle", type=int, default=0, help="If true, take activations only from the top particle.  Otherwise, aggregate them")
    parser.add_argument("--temperature", type=float, default=50, help="Target importance.  Higher values mean activations will jump around more to match the target.")
    parser.add_argument("--saveplots", type=int, default=1, help="Save plots of iterations to disk")
    opt = parser.parse_args()

    if opt.nThreads > 0:
        print("Using {} threads".format(opt.nThreads))
        import torch
        torch.set_num_threads(opt.nThreads)

    pfinal = opt.p
    if opt.pFinal > 0:
        pfinal = opt.pFinal
    assert(pfinal <= opt.p)

    print("Loading corpus audio...")
    tic = time.time()
    (ycorpus, _) = load_corpus(opt.corpus, sr=opt.sr, stereo=(opt.stereo>0))
    print("ycorpus.shape", ycorpus.shape)
    print("Corpus is {:.2f} seconds long".format(ycorpus.shape[1]/opt.sr))
    print("Finished loading up corpus audio: Elapsed Time {:.3f} seconds".format(time.time()-tic))

    feature_params = dict(
        hop=opt.hopLength,
        sr=opt.sr,
        min_freq=opt.minFreq,
        max_freq=opt.maxFreq,
        bins_per_octave=opt.binsPerOctave
    )
    particle_params = dict(
        p=opt.p,
        pfinal=pfinal,
        pd=opt.pd,
        temperature=opt.temperature,
        L=opt.L,
        P=opt.particles,
        r=opt.r,
        neff_thresh=0.1*opt.particles,
        alpha=opt.alpha,
        use_top_particle=opt.useTopParticle == 1,
        max_shift=opt.maxShift
    )
    couple_channels = opt.stereo < 2
    pf = ParticleAudioProcessor(ycorpus, feature_params, particle_params, opt.device, opt.target=="mic", couple_channels)
    ytarget = load_corpus(opt.target, sr=opt.sr, stereo=(opt.stereo>0))
    tic = time.time()
    pf.process_audio_offline(ytarget)
    print("Elapsed time offline particle filter: {:.3f}".format(time.time()-tic))
    if opt.saveplots == 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pf.plot_statistics()
        plt.tight_layout()
        plt.savefig("Stats.svg", bbox_inches='tight')
