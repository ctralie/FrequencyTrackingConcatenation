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


import numpy as np
import matplotlib.pyplot as plt
from probutils import stochastic_universal_sample, do_KL, do_KL_torch, get_activations_diff_sparse, get_diag_lengths_sparse
from observer import Observer
from propagator import Propagator
from audioutils import get_cqt, hann_window, get_windowed
import time

CORPUS_DB_CUTOFF = -50

class ParticleFilterChannel:
    """
    A particle filter for one channel
    """
    def reset_particles(self):
        """
        Randomly reset particles, each with the same weight 1/P
        """
        self.ws = np.array(np.ones(self.P)/self.P, dtype=np.float32) # Particle weights
        if self.device == "np":
            self.states = np.random.randint(self.N, size=(self.P, self.p))
            self.shifts = np.zeros(shape=(self.P, self.p), dtype=int)
        else:
            import torch
            self.ws = torch.from_numpy(self.ws).to(self.device) 
            self.states = torch.randint(self.N, size=(self.P, self.p), dtype=torch.int32).to(self.device) # Particles
            self.shifts = torch.randint(-self.max_shift, self.max_shift+1, size=(self.P, self.p), dtype=torch.int32).to(self.device) # Pitch shifts

    def reset_state(self):
        self.neff = [] # Number of effective particles over time
        self.wsmax = [] # Max weights over time
        self.ws = [] # Weights over time
        self.topcounts = [] 
        self.chosen_idxs = [] # Keep track of chosen indices
        self.chosen_shifts = [] # Keep track of the chosen pitch shifts in a parallel array
        self.H = [] # Activations of chosen indices
        self.reset_particles()
        self.all_ws = []
        self.fit = 0 # KL fit
        self.num_resample = 0

    def __init__(self, ycorpus, corpus_labels, feature_params, particle_params, device, name="channel"):
        """
        ycorpus: ndarray(n_samples)
            Audio samples for the corpus for this channel
        corpus_labels: ndarray(n_samples//hop)
            Labels of each corpus window
        feature_params: {
            hop: int
                Hop length for CQT
            sr: int
                Audio sample rate
            min_freq: float
                Minimum frequency to use (in hz)
            max_freq: float
                Maximum frequency to use (in hz)
            bins_per_octave: int
                Number of CQT bins per octave,
            max_shift: int
                Maximum CQT bins to shift up or down
        }
        particle_params: {
            p: int
                Sparsity parameter for particles
            pfinal: int
                Sparsity parameter for final activations
            pd: float
                State transition probability
            temperature: float
                Amount to focus on matching observations
            L: int
                Number of iterations for NMF observation probabilities
            P: int
                Number of particles
            r: int
                Repeated activations cutoff
            neff_thresh: float
                Number of effective particles below which to resample
            alpha: float
                L2 penalty for weights
            use_top_particle: bool
                If True, only take activations from the top particle at each step.
                If False, aggregate 
        }
        device: string
            Device string for torch
        name: string
            Name for this channel
        """
        tic = time.time()
        self.name = name
        hop = feature_params["hop"]
        sr = feature_params["sr"]
        self.p = particle_params["p"]
        self.P = particle_params["P"]
        self.pfinal = particle_params["pfinal"]
        self.pd = particle_params["pd"]
        self.temperature = particle_params["temperature"]
        self.L = particle_params["L"]
        self.r = particle_params["r"]
        self.neff_thresh = particle_params["neff_thresh"]
        self.alpha = particle_params["alpha"]
        self.use_top_particle = particle_params["use_top_particle"]
        self.max_shift = feature_params["max_shift"]
        self.device = device
        self.sr = sr
        self.hop = hop
        # Store other channels whose parameters are coupled to this channel
        self.coupled_channels = [] 

        ## Step 1: Compute CQT features for corpus
        print("Computing corpus features for {}...".format(name), flush=True)
        WCorpus, WPowers = get_cqt(ycorpus, feature_params, max_shift=self.max_shift)
        self.WCorpus = WCorpus
        # Shrink elements that are too small
        self.WAlpha = self.alpha*np.array(WPowers <= CORPUS_DB_CUTOFF, dtype=np.float32)
        if self.device != "np":
            import torch
            self.WCorpus = torch.from_numpy(self.WCorpus).to(self.device)
            self.WAlpha = torch.from_numpy(self.WAlpha).to(self.device)
        self.loud_enough_idx_map = np.arange(WCorpus.shape[1])[WPowers > CORPUS_DB_CUTOFF]
        print("{:.3f}% of corpus in {} is above loudness threshold".format(100*self.loud_enough_idx_map.size/WCorpus.shape[1], name))
        
        ## Step 2: Setup observer and propagator
        N = WCorpus.shape[1]
        self.N = N
        self.observer = Observer(self.p, self.WCorpus, self.WAlpha, self.L, self.temperature, self.max_shift, device)
        self.propagator = Propagator(corpus_labels, self.pd, self.max_shift, device)
        self.reset_state()

        print("Finished setting up particle filter for {}: Elapsed Time {:.3f} seconds".format(name, time.time()-tic))

    def get_H(self, sparse=False):
        """
        Convert chosen_idxs and H into a numpy array with 
        activations in the proper indices

        Parameters
        ----------
        sparse: bool
            If True, return the sparse matrix directly

        Returns
        -------
        H: ndarray(N, T)
            Activations of the corpus over time
        """
        from scipy import sparse
        N = self.WCorpus.shape[1]
        T = len(self.H)
        vals = np.array(self.H).flatten()
        print("Min h: {:.3f}, Max h: {:.3f}".format(np.min(vals), np.max(vals)))
        rows = np.array(self.chosen_idxs, dtype=int).flatten()
        cols = np.array(np.ones((1, self.pfinal))*np.arange(T)[:, None], dtype=int).flatten()
        H = sparse.coo_matrix((vals, (rows, cols)), shape=(N, T))
        if not sparse:
            H = H.toarray()
        return H
    
    def aggregate_top_activations(self, diag_fac=10, diag_len=10):
        """
        Aggregate activations from the top weight 0.1*self.P particles together
        to have them vote on the best activations

        Parameters
        ----------
        diag_fac: float
            Factor by which to promote probabilities of activations following
            activations chosen in the last steps
        diag_len: int
            Number of steps to look back for diagonal promotion
        
        Returns
        -------
        idxs: ndarray(pfinal, dtype=int)
            Indices into the corpus of the final best activations chosen
        shifts: ndarray(pfinal, dtype=int)
            An array parallel to idxs which holds the pitch shifts of these activations
        """
        ## Step 1: Aggregate max particles
        PTop = int(self.neff_thresh)
        N = self.WCorpus.shape[1]
        ws = self.ws
        if self.device != "np":
            ws = ws.cpu().numpy()
        idxs = np.argpartition(-ws, PTop)[0:PTop]
        states = self.states[idxs, :]
        shifts = self.shifts[idxs, :]
        if self.device != "np":
            states = states.cpu().numpy()
            shifts = shifts.cpu().numpy()
        ws = ws[idxs]
        probs = {}
        for wi, statesi, shiftsi in zip(ws, states, shifts):
            for idx, s in zip(statesi, shiftsi):
                if not (idx, s) in probs:
                    probs[(idx, s)] = wi
                else:
                    probs[(idx, s)] += wi
        
        ## Step 2: Promote states that follow the last state that was chosen
        promoted_idxs = set([])
        for dc in range(1, min(diag_len, len(self.chosen_idxs))+1):
            last_states = self.chosen_idxs[-dc]+dc
            last_shifts = self.chosen_shifts[-dc][last_states < N]
            last_states = last_states[last_states < N]
            for idx, s in zip(last_states, last_shifts):
                if not (idx, s) in promoted_idxs:
                    if (idx, s) in probs:
                        probs[(idx, s)] *= diag_fac
                    promoted_idxs.add((idx, s))

        ## Step 3: Zero out activations that happened over the last
        # r steps prevent repeated activations
        for dc in range(1, min(self.r, len(self.chosen_idxs))+1):
            for idx, s in zip(self.chosen_idxs[-dc], self.chosen_shifts[-dc]):
                if (idx, s) in probs:
                    probs.pop((idx, s))
        
        ## Step 4: Choose top pfinal corpus activations
        items = list(probs.items())
        idxs = np.array([k[0][0] for k in items], dtype=int)
        shifts = np.array([k[0][1] for k in items], dtype=int)
        if idxs.size <= self.pfinal:
            # In some rare cases, we don't have enough activations to choose from
            # so we have to duplicate some of them
            if idxs.size == 0:
                # In the incredibly rare case that all the weights were 0
                # simply choose random windows with zero shifts
                idxs = np.random.randint(N, size=(self.pfinal,))
                shifts = np.zeros(self.pfinal, dtype=int)
            else:
                while idxs.size < self.pfinal:
                    idxs = np.concatenate((idxs, idxs))[0:self.pfinal]
                    shifts = np.concatenate((shifts, shifts))[0:self.pfinal]
        else:
            # Common case: Choose top pfinal corpus positions by weight
            vals = np.array([k[1] for k in items])
            top_idxs = np.argpartition(-vals, self.pfinal)[0:self.pfinal]
            idxs = idxs[top_idxs]
            shifts = shifts[top_idxs]
        return idxs, shifts

    def do_particle_step(self, Vt):
        """
        Run the particle filter for one step given the audio
        in one full window for this channel, and figure out what
        the best activations are

        Vt: ndarray or torch (n_fft, 1)
            Spectrogram at this time

        Returns
        -------
        top_idxs: ndarray(pfinal, dtype=int)
            Indices into the corpus of the final best activations chosen
        top_shifts: ndarray(pfinal, dtype=int)
            An array parallel to idxs which holds the pitch shifts of these activations
        """
        ## Step 1: Propagate
        self.propagator.propagate(self.states, self.shifts)

        ## Step 2: Apply the observation probability updates
        self.ws *= self.observer.observe(self.states, self.shifts, Vt)

        ## Step 3: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        if self.device == "np":
            self.wsmax.append(np.max(self.ws))
        else:
            import torch
            self.wsmax.append(torch.max(self.ws).item())
        if self.use_top_particle:
            selection = torch.argmax(self.ws)
            top_idxs = self.states[selection, :]
            top_shifts = self.shifts[selection, :]
        else:
            top_idxs, top_shifts = self.aggregate_top_activations()
        self.chosen_idxs.append(top_idxs)
        self.chosen_shifts.append(top_shifts)
        
        ## Step 4: Resample particles if effective number is too low
        if self.device == "np":
            self.ws /= np.sum(self.ws)
            self.all_ws.append(np.array(self.ws))
            self.neff.append(1/np.sum(self.ws**2))
        else:
            import torch
            self.ws /= torch.sum(self.ws)
            self.all_ws.append(self.ws.cpu().numpy())
            self.neff.append((1/torch.sum(self.ws**2)).item())
        if self.neff[-1] < self.neff_thresh:
            ## TODO: torch-ify stochastic universal sample
            self.num_resample += 1
            choices, _ = stochastic_universal_sample(self.all_ws[-1], len(self.ws))
            choices = np.array(choices, dtype=int)
            if self.device != "np":
                import torch
                choices = torch.from_numpy(choices).to(self.device)
            self.states = self.states[choices, :]
            self.shifts = self.shifts[choices, :]
            if self.device == "np":
                self.ws = np.ones(self.ws.shape)/self.ws.size
            else:
                import torch
                self.ws = torch.ones(self.ws.shape).to(self.ws)/self.ws.numel()

        return top_idxs, top_shifts
    
    def fit_activations(self, Vt, idxs, shifts):
        """
        Fit activations and mix audio

        Parameters
        ----------
        Vt: ndarray or torch (n_bins, 1)
            CQT frame at this time
        idxs: ndarray(p, dtype=int)
            Indices of activations to use
        shifts: ndarray(p, dtype=int)
            Shifts of these activations
        
        Returns
        -------
        h: ndarray(p)
            Resulting weights
        """
        ## Step 1: Compute activation weights
        kl_fn = do_KL
        if self.device != "np":
            kl_fn = do_KL_torch
        Wi = self.WCorpus[:, idxs]
        CN = Wi.shape[0] - 2*self.max_shift
        idx = np.arange(self.max_shift, self.max_shift+CN)[:, None] - shifts[None, :]
        Wi = np.take_along_axis(Wi, idx, axis=0) # Take properly pitch-shifted features
        h = kl_fn(Wi, self.WAlpha[idxs], Vt[:, 0], self.L)
        hnp = h
        if self.device != "np":
            hnp = h.cpu().numpy()
        self.H.append(hnp)

        ## Step 2: Accumulate KL term for fit
        if self.device == "np":
            WH = Wi.dot(h)
        else:
            from torch import matmul
            WH = matmul(Wi, h)
        Vt = Vt.flatten()
        # Take care of numerical issues
        Vt = Vt[WH > 0]
        WH = WH[WH > 0]
        WH = WH[Vt > 0]
        Vt = Vt[Vt > 0]
        if self.device == "np":
            kl = np.sum(Vt*np.log(Vt/WH) - Vt + WH)
        else:
            import torch
            kl = (torch.sum(Vt*torch.log(Vt/WH) - Vt + WH)).item()
        self.fit += kl

        return hnp

class ParticleAudioProcessor:
    """
    A class that has the following responsibilities:
        * Handles input/output, possibly using the microphone
        * Coordinates particle filters for each channel
        * Creates an over-arching GUI when doing real time input
        * Provides a method to plot statistics of the particle filters
          for each channel once a run has finished
    """
    def reset_state(self):
        """
        Reset all of the audio buffers and particle filters
        """
        # Keep track of time to process each frame
        self.frame_times = [] 
        for c in self.channels:
            c.reset_state()

    def __init__(self, ycorpus, start_idxs, feature_params, particle_params, device, couple_channels=True):
        """
        ycorpus: ndarray(n_channels, n_samples)
            Audio samples for the corpus, possibly multi-channel
        start_idxs: list of n_samples//hop
            Start of each file in the corpus, in units of hop
        feature_params: {
            hop: int
                Hop length for CQT
            sr: int
                Audio sample rate
            min_freq: float
                Minimum frequency to use (in hz)
            max_freq: float
                Maximum frequency to use (in hz)
            bins_per_octave: int
                Number of CQT bins per octave
            max_shift: int
                Maximum CQT bins to shift up or down
        }
        particle_params: {
            p: int
                Sparsity parameter for particles
            pfinal: int
                Sparsity parameter for final activations
            pd: float
                State transition probability
            temperature: float
                Amount to focus on matching observations
            L: int
                Number of iterations for NMF observation probabilities
            P: int
                Number of particles
            r: int
                Repeated activations cutoff
            neff_thresh: float
                Number of effective particles below which to resample
            alpha: float
                L2 penalty for weights
            use_top_particle: bool
                If True, only take activations from the top particle at each step.
                If False, aggregate 
        }
        device: string
            Device string for torch
        couple_channels: bool
            If true, only run a particle filter on the first channel, and use the
            same corpus elements on the other channels. (Default true)
            Otherwise, run individual particle filters on each channel
        """
        self.device = device
        self.couple_channels = couple_channels
        self.feature_params = feature_params
        self.sr = feature_params["sr"]
        self.hop = feature_params["hop"]
        self.n_channels = ycorpus.shape[0]
        self.ycorpus = ycorpus
        ## Compute an indicator vector of which file each corpus element is in
        N = ycorpus.shape[1] // self.hop
        corpus_labels = np.zeros(N, dtype=int)
        for i in range(len(start_idxs)-1):
            corpus_labels[start_idxs[i]:start_idxs[i+1]] = i
        self.corpus_labels = corpus_labels
        self.channels = [ParticleFilterChannel(ycorpus[i, :], corpus_labels, feature_params, particle_params, device, name="channel {}".format(i)) for i in range(ycorpus.shape[0])]
        if self.couple_channels:
            for c in self.channels[1:]:
                self.channels[0].coupled_channels.append(c)
        self.reset_state()

    def process_audio_offline(self, ytarget):
        """
        Process audio audio offline, frame by frame

        Parameters
        ----------
        ytarget: ndarray(n_channels, T)
            Audio samples to process

        Returns
        -------
        ndarray(n_samples, n_channels)
            Generated audio
        """
        from tqdm import tqdm
        if len(ytarget.shape) == 1:
            ytarget = ytarget[None, :] # Mono audio
        n_channels = ytarget.shape[0]

        ## Step 1: Compute CQT on each channel
        CTarget = [get_cqt(ytarget[i, :], self.feature_params)[0] for i in range(n_channels)]

        ## Step 2: Run each CQT frame through the particle filter
        hop = self.hop
        y = np.zeros((self.n_channels, CTarget[0].shape[1]*hop+hop), dtype=np.float32)
        hann_samples = hann_window(hop*2)
        for t in tqdm(range(CTarget[0].shape[1])):
            tic = time.time()
            idxs = []
            shifts = []
            for i, (c, V) in enumerate(zip(self.channels, CTarget)):
                # Run each particle filter on its channel of audio
                Vt = V[:, t][:, None]
                if self.device != "np":
                    import torch
                    Vt = torch.from_numpy(Vt).to(self.device)
                if i == 0 or not self.couple_channels:
                    idxs, shifts = c.do_particle_step(Vt)
                h = c.fit_activations(Vt, idxs, shifts)
                yt = np.zeros(hop*2)
                for hi, idx, shift in zip(h, idxs, shifts):
                    if shift == 0:
                        yti = self.ycorpus[i, idx*hop:idx*hop+2*hop]
                    else:
                        from pyrubberband import pyrb
                        ## Take a half a second of audio for enough context to shift
                        chunk = self.sr//4
                        yti = self.ycorpus[i, idx*hop:idx*hop+chunk]
                        if yti.size < chunk:
                            yti = np.concatenate((yti, np.zeros(chunk-yti.size)))
                        yti = pyrb.pitch_shift(yti, self.sr, shift)
                        yti = yti[0:hop*2]
                    if yti.size < 2*hop:
                        yti = np.concatenate((yti, np.zeros(2*hop-yti.size)))
                    yt += hi*yti
                y[i, t*hop:t*hop+2*hop] += hann_samples*yt
            # Record elapsed time
            elapsed = time.time()-tic
            self.frame_times.append(elapsed)
        return y


    def plot_statistics(self):
        """
        Plot statistics about the activations that were chosen
        """
        p = self.channels[0].states.shape[1]
        channels_to_plot = self.channels
        if self.couple_channels:
            channels_to_plot = channels_to_plot[0:1]
        Hs = [c.get_H(sparse=True) for c in channels_to_plot]
        all_active_diffs = [get_activations_diff_sparse(H.row, H.col, p) for H in Hs]
        
        plt.subplot2grid((2, 3), (0, 0), colspan=2)
        legend = []
        for (active_diffs, c) in zip(all_active_diffs, channels_to_plot):
            t = np.arange(active_diffs.size)*self.hop/(self.sr)
            plt.plot(t, active_diffs, linewidth=0.5)
            legend.append("{}: Mean {:.3f}".format(c.name, np.mean(active_diffs)))
        plt.legend(legend)
        plt.title("Activation Changes over Time, p={}".format(p))
        plt.xlabel("Time (Seconds)")

        plt.subplot(233)
        legend = []
        for (active_diffs, c) in zip(all_active_diffs, channels_to_plot):
            plt.hist(active_diffs, bins=np.arange(p+2), alpha=0.5)
            legend.append(c.name)
        plt.legend(legend)
        plt.title("Activation Changes Histogram")
        plt.xlabel("Number of Activations Changed")
        plt.ylabel("Counts")

        plt.subplot(234)
        legend = []
        for c in channels_to_plot:
            plt.plot(c.wsmax)
            legend.append("{}: {:.2f}".format(c.name, c.fit))
        plt.legend(legend)
        plt.title("Max Probability And Overall Fit")
        plt.xlabel("Timestep")

        plt.subplot(235)
        legend = []
        for c in channels_to_plot:
            plt.plot(c.neff)
            legend.append("{} Med {:.2f}, Resampled {}x".format(c.name, np.median(c.neff), c.num_resample))
        plt.legend(legend)
        plt.xlabel("Timestep")
        plt.title("Neff (P={})".format(channels_to_plot[0].P))

        plt.subplot(236)
        legend = []
        for c, H in zip(channels_to_plot, Hs):
            diags = get_diag_lengths_sparse(H.row, H.col)
            legend.append("{} Mean: {:.3f}".format(c.name, np.mean(diags)))
            plt.hist(diags, bins=np.arange(30), alpha=0.5)
        plt.legend(legend)
        plt.xlabel("Diagonal Length ($p_d$={})".format(channels_to_plot[0].pd))
        plt.ylabel("Counts")
        plt.title("Diagonal Lengths (Temperature {})".format(channels_to_plot[0].temperature))