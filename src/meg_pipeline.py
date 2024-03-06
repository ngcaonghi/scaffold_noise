import numpy as np
import hcp
import mne
import os
import glob
import gc
from mne.beamformer import make_lcmv, apply_lcmv_raw, apply_lcmv_epochs
from mne import extract_label_time_course
from mne.filter import notch_filter
from neurodsp.aperiodic import compute_irasa
from .utils import WrappedPartial, FunctionNode
from functools import partial

# exports
__all__ = [
    'source_loc_ingredients', 
    'read_preproc_mats', 
    'remove_bad', 
    'apply_ica', 
    'compute_source_estimate', 
    'parcellate', 
    'rearrange_meg', 
    'roi_irasa', 
    'meg_pipeline',
    'make_bandpower_ratio',
    'create_bands'
]

NUM_EPOCHS = 2
EPOCH_DURATION = 12 # seconds
TMIN = 0
TMAX = EPOCH_DURATION
SEED = 42

# Pipeline functions ============================================================

def _write_bem_solution( 
        subject,
        subjects_dir,
        recordings_path,
        hcp_path,
        surface,
        spacing,
        add_dist,
        src_type
    ):
    """
    Write BEM solution for MEG pipeline.

    Args:
        - subject (str): The subject identifier.
        - subjects_dir (str): The directory where subject-specific data is stored.
        - recordings_path (str): The path to the recordings.
        - hcp_path (str): The path to the HCP data.
        - surface (str): The surface type.
        - spacing (float): The spacing parameter.
        - add_dist (float): The additional distance parameter.
        - src_type (str): The source type.

    Returns:
        - dict: The output dictionary containing the computed forward stack, source spaces, and BEM solution.
    """
    if type(subject) != str:
        subject = str(subject)
    hcp.make_mne_anatomy(
        subject=subject, 
        subjects_dir=subjects_dir,
        hcp_path=hcp_path,
        recordings_path=recordings_path,
        outputs=('label', 'mri', 'stats', 'surf', 'touch')
    )
    out = hcp.compute_forward_stack(
        subjects_dir=subjects_dir, 
        subject=subject,
        recordings_path=recordings_path,
        hcp_path=hcp_path,
        src_params=dict(spacing=spacing,
        add_dist=add_dist, 
        surface=surface)
    )
    mne.write_forward_solution(
        os.path.join(
            recordings_path, subject, '%s-%s-%s-%s-fwd.fif' % (
                surface, spacing, add_dist, src_type
            )
        ),
        out['fwd'], 
        overwrite=True
    )
    mne.write_source_spaces(
        os.path.join(recordings_path, subject, '%s-%s-%s-%s-src.fif' % (
                surface, spacing, add_dist, src_type
            )
        ),
        out['src_subject']
    )
    fname_bem = os.path.join(
        subjects_dir, subject, 'bem', '%s-%i-bem.fif' % (
            subject, out['bem_sol']['solution'].shape[0]
        )
    )
    mne.write_bem_solution(fname_bem, out['bem_sol'])
    return out


def _read_source_ingredients(
        subject, 
        subjects_dir,
        hcp_path,
        label_folder
    ):
    """
    Read the necessary ingredients for source analysis from the given subject.

    Args:
        - subject (str): The subject identifier.
        - subjects_dir (str): The path to the subjects directory.
        - hcp_path (str): The path to the HCP data.
        - label_folder (str): The folder containing the label files.

    Returns:
        - tuple: A tuple containing the labels, forward solution, and source space.
    """
    subject = str(subject)
    labels = [mne.read_label(f, subject) for f in glob.glob(subjects_dir+label_folder+subject+'/label/*.label')]
    if len(labels) > 0:
        raise ValueError("No atlas labels found")
    fwd = mne.read_forward_solution(hcp_path + subject + "/white-oct6-True-subject_on_fsaverage-fwd.fif")
    source_space = mne.read_source_spaces(hcp_path + subject + '/white-oct6-True-subject_on_fsaverage-src.fif')
    return labels, fwd, source_space


def source_loc_ingredients(
        subject, 
        hcp_path, 
        recordings_path, 
        subjects_dir, 
        label_folder,
        surface,
        spacing,
        add_dist,
        src_type
    ):
    """
    Prepare ingredients for source localization.

    Args:
        - subject (str): Subject identifier.
        - hcp_path (str): Path to the HCP dataset.
        - recordings_path (str): Path to the recordings.
        - subjects_dir (str): Path to the subjects directory.
        - label_folder (str): Folder containing labels.
        - surface (str): Surface type.
        - spacing (float): Spacing value.
        - add_dist (bool): Flag indicating whether to add distance.
        - src_type (str): Source type.

    Returns:
        - dict: Dictionary containing the following ingredients:
            - 'bem': BEM solution.
            - 'labels': Labels.
            - 'fwd': Forward solution.
            - 'source_space': Source space.
            - 'noise_cov': Noise covariance.
    """
    bem = _write_bem_solution(
        subject,
        subjects_dir,
        recordings_path,
        hcp_path,
        surface,
        spacing,
        add_dist,
        src_type
    )
    labels, fwd, source_space = _read_source_ingredients(
        subject, subjects_dir, hcp_path, label_folder
    )
    noise = hcp.read_raw(subject, 'noise_subject', hcp_path=hcp_path)
    noise_cov = mne.compute_raw_covariance(noise)
    del noise; gc.collect()
    ingredients = {
        'labels': labels,
        'fwd': fwd,
        'source_space': source_space,
        'noise_cov': noise_cov
    }
    return ingredients


def read_preproc_mats(subject, task, hcp_path, ingredients):
    """
    Read preprocessed materials for a given subject and task.

    Args:
        - subject (str): The subject identifier.
        - task (str): The task identifier.
        - hcp_path (str): The path to the HCP dataset.
        - ingredients (list): A list of ingredients.

    Returns:
        - dict: A dictionary containing the following preprocessed materials:
            - 'raw': The raw data.
            - 'annots': The annotations.
            - 'ica': The independent component analysis.
            - 'ingredients': The list of ingredients.
    """
    raw = hcp.read_raw(subject, task, hcp_path=hcp_path)
    raw.pick(["meg"])
    annots = hcp.read_annot(subject, task, hcp_path=hcp_path)
    ica = hcp.read_ica(subject, task, hcp_path=hcp_path)
    return {'raw': raw, 'annots': annots, 'ica': ica, 'ingredients': ingredients}


def remove_bad(raw, annots, ica, ingredients):
    """
    Remove bad segments and channels from the raw data.

    Args:
        - raw (mne.io.Raw): The raw data.
        - annots (dict): Dictionary containing annotations.
        - ica (mne.preprocessing.ICA): The ICA object.
        - ingredients (list): List of ingredients.

    Returns:
        - dict: A dictionary containing
            - 'raw': The raw data with bad segments and channels removed.
            - 'exclude': The indices of the independent components to be excluded.
            - 'ica': The ICA object.
            - 'ingredients': The list of ingredients.
    """
    bad_seg = (annots['segments']['all']) / raw.info['sfreq']
    annotations = mne.Annotations(
        bad_seg[:, 0], 
        (bad_seg[:, 1] - bad_seg[:, 0]), 
        description='bad'
    )
    raw = raw.set_annotations(annotations)
    raw.info['bads'].extend(annots['channels']['all'])
    exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
                           if ii not in annots['ica']['brain_ic_vs']]
    return {'raw': raw, 'exclude': exclude, 'ica': ica, 'ingredients': ingredients}


def apply_ica(raw, ica, exclude, ingredients):
    """
    Apply independent component analysis (ICA) to the raw data.

    Args:
        - raw (Raw): The raw data to apply ICA to.
        - ica (ndarray): The ICA matrix.
        - exclude (list): List of component indices to exclude.
        - ingredients (dict): Dictionary of ingredients.

    Returns:
        - dict: A dictionary containing
            - 'raw': The raw data with ICA applied.
            - 'ingredients': The list of ingredients.
    """
    hcp.preprocessing.apply_ica_hcp(raw.load_data(), ica_mat=ica, exclude=exclude)
    return {'raw': raw, 'ingredients': ingredients}


def _compute_source_estimate_task(raw, noise_cov, forward, tmin, tmax, baseline, preload, num_epochs):
    """
    Compute the source estimate using the linearly constrained minimum variance (LCMV) beamformer.

    Args:
        - raw (mne.io.Raw): The raw MEG data.
        - noise_cov (mne.Covariance): The noise covariance matrix.
        - forward (mne.Forward): The forward solution.
        - tmin (float): The start time of the epoch in seconds.
        - tmax (float): The end time of the epoch in seconds.
        - baseline (tuple or None): The time range to apply baseline correction.
        - preload (bool): Whether to preload the data.
        - num_epochs (int): The number of epochs to subsample.

    Returns:
        - source_estimate (mne.SourceEstimate): The computed source estimate.
    """

    events = mne.find_events(raw)
    for event_id in events:
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=tmin, tmax=tmax, baseline=baseline, preload=preload
        )
        subsample_idx = np.random.choice(len(epochs), num_epochs, replace=False)
        epochs = epochs[subsample_idx]
        data_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method="empirical")
        filters = make_lcmv(
            epochs.info,
            forward,
            data_cov,
            reg=0.05,
            noise_cov=noise_cov,
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            rank=None,
        )
        source_estimate = apply_lcmv_epochs(epochs, filters)
    return source_estimate


def _compute_source_estimate_rest(raw, noise_cov, forward, duration):
    """
    Compute the source estimate for resting state data.

    Args:
        - raw (mne.io.Raw): The raw MEG data.
        - noise_cov (mne.Covariance): The noise covariance matrix.
        - forward (mne.Forward): The forward solution.
        - duration (float): The duration of the data segment in seconds.

    Returns:
        - source_estimate (mne.SourceEstimate): The computed source estimate.
    """

    data_cov = mne.compute_raw_covariance(raw, method="empirical")
    filters = make_lcmv(
        raw.info,
        forward,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None,
    )
    # choose random start and stop points that are [duration] seconds apart
    start = np.random.randint(0, len(raw) - int(duration * raw.info['sfreq']))
    stop = start + int(duration * raw.info['sfreq'])
    source_estimate = apply_lcmv_raw(raw, filters, start=start, stop=stop)
    return source_estimate


def compute_source_estimate(
        raw, ingredients, task, tmin, tmax, 
        baseline,
        preload,
        num_epochs,
        duration
    ):
    """
    Compute the source estimate for the given raw data and ingredients.

    Args:
        - raw (mne.io.Raw): The raw data.
        - ingredients (dict): A dictionary containing the necessary ingredients for computation.
        - task (str): The task name.
        - tmin (float): The start time of the time window.
        - tmax (float): The end time of the time window.
        - baseline (tuple, optional): The baseline period. Defaults to (None, 0).
        - preload (bool, optional): Whether to preload the data. Defaults to True.
        - num_epochs (int, optional): The number of epochs. Defaults to NUM_EPOCHS.
        - duration (float, optional): The duration of the epochs. Defaults to EPOCH_DURATION * NUM_EPOCHS.

    Returns:
        - dict: A dictionary containing
            - 'source_estimate': The computed source estimate.
            - 'labels': The labels.
            - 'source_space': The source space.
            - 'sfreq': The sampling frequency.
    """
    noise_cov = ingredients['noise_cov']
    fwd = ingredients['fwd']
    if task != 'rest':
        source_estimate = _compute_source_estimate_task(
            raw, noise_cov, fwd, tmin, tmax, baseline, preload, num_epochs
        )
    else:
        source_estimate = _compute_source_estimate_rest(
            raw, noise_cov, fwd, duration
        )
    labels = ingredients['labels']
    source_space = ingredients['source_space']
    return {
        'source_estimate': source_estimate, 
        'labels': labels, 
        'source_space': source_space,
        'sfreq': raw.info['sfreq']
    }


def parcellate(source_estimate, labels, source_space, sfreq):
    """
    Parcellates the source estimate into regions of interest (ROIs) based on the given labels.
    
    Args:
        - source_estimate (array-like): The source estimate data.
        - labels (list): The labels defining the ROIs.
        - source_space (object): The source space object.
        - sfreq (float): The sampling frequency of the data.
    
    Returns:
        - dict: A dictionary containing
            - 'roi_time_series': The time series data for each ROI.
            - 'labels': The labels.
            - 'sfreq': The sampling frequency.
    """
    roi_time_series = extract_label_time_course(
        source_estimate, labels, source_space
    ) # (360, T)
    return {'roi_time_series': roi_time_series, 'labels': labels, 'sfreq': sfreq}


def rearrange_meg(roi_time_series, labels, sfreq, region_names_file):
    """
    Rearranges MEG data based on the given ROI time series, labels, and region names file.
    NOTE: Only implemented for the HCPMMP1 atlas. Download the atlas here: https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446
    
    Args:
        - roi_time_series (ndarray or list): The ROI time series data.
        - labels (list): The labels corresponding to the MEG data.
        - sfreq (float): The sampling frequency of the MEG data.
        - region_names_file (str): The file path to the region names file.
        
    Returns:
        - dict: A dictionary containing the rearranged MEG data and the sampling frequency.
            - 'X_list' (list): The rearranged MEG data.
            - 'sfreq' (float): The sampling frequency.
    """
    # XXX Only implemented for the HCPMMP1 atlas
    mri_region_names = np.load(region_names_file)
    meg_mri_map = {}
    for imeg, rmeg in enumerate(labels):
        for imri, rmri in enumerate(mri_region_names):
            if rmri[0] == 'L':
                rmri += '_ROI-lh'
                pref = 'lh.'
            else:
                rmri += '_ROI-rh'
                pref = 'rh.'
            if rmeg.name == pref+rmri:
                meg_mri_map[imeg] = imri
    if not isinstance(roi_time_series, list):
        roi_time_series = [roi_time_series]
    new_arr = [np.zeros((360, roi_time_series[0].shape[-1])) for _ in range(len(roi_time_series))]
    for i, epoch in enumerate(roi_time_series):
        for imeg, imri in meg_mri_map.items():
            new_arr[i][imri] = epoch[imeg, :]
    return {
        'X_list': new_arr, 
        'sfreq' : sfreq
    }


def roi_irasa(X_list, sfreq, fmin, fmax, bandstops, notch_widths, **spectrum_kwargs):
    """
    Compute the Region of Interest (ROI) Integrated Resting-State Amplitude (IRASA) for multiple signals.

    Args:
        - X_list (list): List of signals, each represented as a 1D numpy array.
        - sfreq (float): Sampling frequency of the signals.
        - fmin (float): Minimum frequency of interest for IRASA computation.
        - fmax (float): Maximum frequency of interest for IRASA computation.
        - bandstops (list): List of tuples specifying the frequency ranges to be notch filtered.
        - notch_widths (list): List of notch filter widths corresponding to each bandstop range.
        - **spectrum_kwargs: Additional keyword arguments to be passed to the compute_irasa function.
        See documentation for neurodsp.aperiodic.compute_irasa for more details.

    Returns:
        - dict: A dictionary containing the computed IRASA results:
            - 'psds_aperiodic' (ndarray): Mean aperiodic power spectra across all ROIs.
            - 'psds_periodic' (ndarray): Mean periodic power spectra across all ROIs.
            - 'freqs' (ndarray): Frequency values corresponding to the power spectra.
    """
    psds_aperiodic, psds_periodic = [], []   
    if 'nperseg' in spectrum_kwargs \
        and isinstance(spectrum_kwargs['nperseg'], str) \
        and spectrum_kwargs['nperseg'][-1] == 's':
        spectrum_kwargs['nperseg'] = int(spectrum_kwargs['nperseg'][:-1] * sfreq)
    if 'noverlap' in spectrum_kwargs \
        and isinstance(spectrum_kwargs['noverlap'], str) \
        and spectrum_kwargs['noverlap'][-1] == 's':
        spectrum_kwargs['noverlap'] = int(spectrum_kwargs['noverlap'][:-1] * sfreq)
    for X in X_list:
        X = notch_filter(X, sfreq, bandstops, notch_widths)
        psds_aperiodic_rois, psds_periodic_rois = [], []
        for roi in X:
            freqs, apsd, ppsd = compute_irasa(
                roi, sfreq, (fmin, fmax), **spectrum_kwargs
            )
            psds_aperiodic_rois.append(apsd)
            psds_periodic_rois.append(ppsd)
        psds_aperiodic.append(psds_aperiodic_rois)
        psds_periodic.append(psds_periodic_rois)
    return {
        'psds_aperiodic': np.array(psds_aperiodic).mean(axis=0), # task-condition average
        'psds_periodic': np.array(psds_periodic).mean(axis=0),
        'freqs': freqs
    }


def meg_pipeline(
        subject, 
        hcp_path, 
        recordings_path, 
        subjects_dir, 
        label_folder,
        region_names_file, 
        surface='white', 
        spacing='oct6', 
        add_dist=True, 
        src_type='subject_on_fsaverage',
        tmin=TMIN,
        tmax=TMAX, 
        baseline=(None, 0),
        num_epochs=NUM_EPOCHS,
        duration=EPOCH_DURATION * NUM_EPOCHS, # total duration of all epochs
        fmin=2, 
        fmax=120, 
        bandstops=[60, 120],
        notch_widths=[1, 1],
        method='welch',
        nperseg='2s',
        noverlap='1s'
    ):
    """
    Perform the MEG pipeline for a given subject.

    Args:
        - subject (str): The subject identifier.
        - hcp_path (str): The path to the HCP data.
        - recordings_path (str): The path to the MEG recordings.
        - subjects_dir (str): The directory where the subject's data will be stored.
        - label_folder (str): The folder containing the labels.
        - region_names_file (str): The file containing the region names.
        - surface (str, optional): The surface to use. Defaults to 'white'.
        - spacing (str, optional): The spacing to use. Defaults to 'oct6'.
        - add_dist (bool, optional): Whether to add distance information. Defaults to True.
        - src_type (str, optional): The source type. Defaults to 'subject_on_fsaverage'.
        - tmin (float, optional): The start time of the analysis. Defaults to TMIN.
        - tmax (float, optional): The end time of the analysis. Defaults to TMAX.
        - baseline (tuple, optional): The baseline period. Defaults to (None, 0).
        - num_epochs (int, optional): The number of epochs. Defaults to NUM_EPOCHS.
        - duration (float, optional): The total duration of all epochs. Defaults to EPOCH_DURATION * NUM_EPOCHS.
        - fmin (int, optional): The minimum frequency for spectral analysis. Defaults to 2.
        - fmax (int, optional): The maximum frequency for spectral analysis. Defaults to 120.
        - bandstops (list, optional): The frequencies to be removed from the spectrum. Defaults to [60, 120].
        - notch_widths (list, optional): The widths of the notches to be applied. Defaults to [1, 1].
        - method (str, optional): The method for spectral analysis. Defaults to 'welch'.
        - nperseg (str, optional): The length of each segment for spectral analysis. Defaults to '2s'.
        - noverlap (str, optional): The overlap between segments for spectral analysis. Defaults to '1s'.

    Returns:
        - FunctionNode: The function tree representing the MEG pipeline.
    """
    # root node
    _source_loc_ingredients = WrappedPartial(
        source_loc_ingredients, 
        subject=subject, 
        hcp_path=hcp_path, 
        recordings_path=recordings_path, 
        subjects_dir=subjects_dir, 
        label_folder=label_folder, 
        surface=surface, 
        spacing=spacing, 
        add_dist=add_dist, 
        src_type=src_type
    ).getfunc('source_loc_ingredients')

    # FUNCTION TREE 
    tree = FunctionNode(_source_loc_ingredients)
    for task in ['task_motor', 'task_working_memory', 'rest']:
        # awaiting input: ingredients
        # return: raw, annots, ica, ingredients
        _read_preproc_mats = WrappedPartial(
            read_preproc_mats, 
            task=task,
            subject=subject,
            hcp_path=hcp_path
        ).getfunc('read_preproc_mats_'+task)
        _read_preproc_mats = FunctionNode(_read_preproc_mats)
        tree.add_child(_read_preproc_mats)

        # awaiting input: raw, annots, ica, ingredients
        # return: raw, exclude, ica, ingredients
        _remove_bad = WrappedPartial(
            remove_bad
        ).getfunc('remove_bad_'+task)
        _remove_bad = FunctionNode(_remove_bad)
        tree.add_child_to_node(_read_preproc_mats, _remove_bad)

        # awaiting input: raw, exclude, ica, ingredients
        # return: raw, ingredients
        _apply_ica = WrappedPartial(
            apply_ica
        ).getfunc('apply_ica_'+task)
        _apply_ica = FunctionNode(_apply_ica)
        tree.add_child_to_node(_remove_bad, _apply_ica)

        # awaiting input: raw, ingredients, task
        # return: source_estimate, labels, source_space, sfreq
        _compute_source_estimate = WrappedPartial(
            compute_source_estimate,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            num_epochs=num_epochs,
            duration=duration
        ).getfunc('compute_source_estimate_'+task)
        _compute_source_estimate = FunctionNode(_compute_source_estimate)
        tree.add_child_to_node(_apply_ica, _compute_source_estimate)

        # awaiting input: source_estimate, labels, source_space, sfreq
        # return: roi_time_series, labels, sfreq
        _parcellate = WrappedPartial(
            parcellate
        ).getfunc('parcellate_'+task)
        _parcellate = FunctionNode(_parcellate)
        tree.add_child_to_node(_compute_source_estimate, _parcellate)

        # awaiting input: roi_time_series, labels, sfreq
        # return: X_list, sfreq
        _rearrange_meg = WrappedPartial(
            rearrange_meg,
            region_names_file=region_names_file
        ).getfunc('rearrange_meg_'+task)
        _rearrange_meg = FunctionNode(_rearrange_meg)
        tree.add_child_to_node(_parcellate, _rearrange_meg)

        # awaiting input: X_list, sfreq
        # return: psds_aperiodic, psds_periodic, freqs
        _roi_irasa = WrappedPartial(
            roi_irasa,
            fmin=fmin,
            fmax=fmax,
            bandstops=bandstops,
            notch_widths=notch_widths,
            method=method,
            nperseg=nperseg,
            noverlap=noverlap
        ).getfunc('roi_irasa_'+task)
        _roi_irasa = FunctionNode(_roi_irasa)
        tree.add_child_to_node(_rearrange_meg, _roi_irasa)

    return tree


# Post-pipeline functions ======================================================

def make_bandpower_ratio(
        psd,
        freqs,
        bands,
        fmin,
        fmax
    ):
    """
    Calculate the bandpower ratio for given power spectral density (psd) and frequency range.

    Args:
        - psd (ndarray): Power spectral density. Can be 1D or 2D.
        - freqs (ndarray): Frequency values.
        - bands (list): List of tuples specifying the frequency bands of interest.
        - fmin (float): Minimum frequency value for the broadband range.
        - fmax (float): Maximum frequency value for the broadband range.

    Returns:
        - dict: Dictionary containing the bandpower ratio for each frequency band.
    """

    if np.ndim(psd) == 1:
        psd = psd[None, :]
    broadband_lim = np.argwhere((freqs >= fmin) & (freqs <= fmax))
    broadband_power = psd[:, broadband_lim].sum(axis=1).squeeze()
    bandpower = {}
    for l, h in bands:
        band_lim = np.argwhere((freqs >= l) & (freqs <= h))
        bandpower = psd[:, band_lim].sum(axis=1).squeeze()
        bandpower[(l, h)] = bandpower / broadband_power
    return bandpower


def create_bands(
        num_points,
        low_freq,
        high_freq,
        scale='log10'
    ):
    """
    Create a list of frequency bands.

    Args:
        - num_points (int, optional): The number of points to use.
        - low_freq (float, optional): The lower frequency bound.
        - high_freq (float, optional): The upper frequency bound.
        - scale (str, optional): The scale to use. Defaults to 'log10'. 
        Can be 'log10', 'log2', 'ln', or 'linear'.

    Returns:
        - list: A list of frequency bands.
    """
    if scale == 'log10':
        sampfunc = np.log10
        sampspace = partial(np.logspace, base=10)
    elif scale == 'log2':
        sampfunc = np.log2
        sampspace = partial(np.logspace, base=2)
    elif scale == 'ln':
        sampfunc = np.log
        sampspace = partial(np.logspace, base=np.e)
    elif scale == 'linear':
        sampfunc = lambda x: x
        sampspace = np.linspace
    else:
        raise ValueError(f"Invalid scale: {scale}")
    log_array = sampspace(
        start = sampfunc(low_freq), 
        stop = sampfunc(high_freq), 
        num = num_points
    )
    bands = list(zip(log_array[:-1], log_array[1:]))
    return bands