import numpy as np
import os
from huggingface_hub import hf_hub_download
import torch
from librosa.filters import mel as librosa_mel_fn
from hydra.utils import instantiate
from omegaconf import DictConfig
import yaml
import torchaudio

def feature_extract(wave16, wave22, ce, se, cfm_len_regulator, dtype):
    mel_fn_args = {
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 80,
        "sampling_rate": 22050,
        "fmin": 0,
        "fmax": None,
        "center": False
    }
    mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
    with torch.autocast(device_type = wave16.device.type, dtype = dtype):
        with torch.no_grad():
            mel = mel_fn(wave22)
            mel_len = mel.size(2)
            
             # Compute content features 
            _, cont, _ = ce(wave16, [wave16.size(-1)], ssl_model = ce.ssl_model)
                # Compute style features
            style = compute_style(wave16, se)
            prompt_condition, _, = cfm_len_regulator(cont, ylens=torch.LongTensor([mel_len]).to(wave16.device.type))
        
    return mel, style, prompt_condition
        


    
def init_model(yaml_path, device, repo_id = None, filename = None, prefix_name = None):
    cfg = DictConfig(yaml.safe_load(open(yaml_path, "r")))
    ext = instantiate(cfg)
    ext.to(device).eval()

    if repo_id is not None and filename is not None:
        checkpoint_path = load_custom_model_from_hf(repo_id = repo_id, model_filename = filename)
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')

        if prefix_name is None:
            ext.load_state_dict(checkpoint, strict = False)
        else:
            dicts = strip_prefix(checkpoint["net"][prefix_name], "module.")
            missing_keys, unexpected_keys = ext.load_state_dict(dicts, strict=False)



    return ext



def compute_style(waves_16k: torch.Tensor, se, wave_lens_16k: torch.Tensor = None):
    if wave_lens_16k is None:
        wave_lens_16k = torch.tensor([waves_16k.size(-1)], dtype=torch.int32).to(waves_16k.device)
    feat_list = []
    for bib in range(waves_16k.size(0)):
        feat = torchaudio.compliance.kaldi.fbank(waves_16k[bib:bib + 1, :wave_lens_16k[bib]],
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat_list.append(feat)
    max_feat_len = max([feat.size(0) for feat in feat_list])
    feat_lens = torch.tensor([feat.size(0) for feat in feat_list], dtype=torch.int32).to(waves_16k.device) // 2
    feat_list = [
        torch.nn.functional.pad(feat, (0, 0, 0, max_feat_len - feat.size(0)), value=float(feat.min().item()))
        for feat in feat_list
    ]
    feat = torch.stack(feat_list, dim=0)
    style = se(feat, feat_lens)
    return style

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints")
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="./checkpoints")

    return model_path, config_path


@staticmethod
def crossfade(chunk1, chunk2, overlap):
    """Apply crossfade between two audio chunks."""
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@staticmethod
def strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    """
    Strip the prefix from the state_dict keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict



mel_basis = {}
hann_window = {}



def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(x, C=1, clip_val=1e-5):
    
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(sampling_rate)}_{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(sampling_rate) + "_" + str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

