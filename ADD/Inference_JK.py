#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:15:00 2025

@author: comm-kwon
"""

import librosa
import torch
from utils import init_model,feature_extract
import time
import scipy.io as sio
import glob
import numpy as np
import yaml
import soundfile as sf

dtypes = {'float32': torch.float32,
          'float16': torch.float16,
          'bfloat16': torch.bfloat16}

dirs = [ "LD004C5",  "LD004C6", "LD004C7"]

for ddir in dirs:
    savedir = "Data/" + ddir + "/"
    with open(savedir + "default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    dtype = dtypes[cfg['dtype']]
    
    se = init_model("configs/style_encoder.yaml", cfg['device'], cfg['SE']['repo_id'], cfg['SE']['checkpoint'])
    cfm_len_regulator = init_model("configs/cfm_length_regulator.yaml", cfg['device'], cfg['default']['repo_id'], cfg['default']['CFM_CHECKPOINT'], prefix_name = "length_regulator")
    cfm = init_model("configs/cfm.yaml", cfg['device'], cfg['default']['repo_id'], cfg['default']['CFM_CHECKPOINT'], prefix_name = "cfm")
    vocoder = init_model("configs/vocoder.yaml", cfg['device'])
    ce = init_model("configs/context_extractor.yaml", cfg['device'], cfg['CE']['repo_id'], cfg['CE']['wide_checkpoint'])
    
    target_wave22 = torch.tensor(librosa.load(cfg['target_audio_path'], sr=22050)[0]).unsqueeze(0).float().to(cfg['device'])
    target_wave16 = torch.tensor(librosa.load(cfg['target_audio_path'], sr=16000)[0]).unsqueeze(0).float().to(cfg['device'])
    
    tar_mel, tar_style, tar_cond = feature_extract(target_wave16, target_wave22, ce, se, cfm_len_regulator, dtype)
    max_context_window = cfg['sr'] // cfg['hop_size'] * cfg['max_context_len']
    overlap_wave_len = cfg['overlap_frame_len'] * cfg['hop_size']
            
    
    max_source_window = max_context_window - tar_mel.size(2)
    
    filepath = 'Data/LD004B'
    files = glob.glob(filepath + '/*.wav')
    files.sort()
    
    
    audlen = np.zeros((len(files),2))
    
    for i in range(len(files)):
        filename = files[i]
        s = time.time()
        #source_audio_path = 'Data/LD004B/' + filename
        source_wave_all = librosa.load(filename, sr=16000)[0]
        
        audlen[i,0] = source_wave_all.shape[0]/16000
        
        
        source_wave = source_wave_all
        source_wave16 = torch.FloatTensor(source_wave).unsqueeze(0).to(cfg['device'])
        source_wave22 = torch.FloatTensor(librosa.resample(y = source_wave, target_sr = 22050, orig_sr = 16000)).unsqueeze(0).to(cfg['device'])
            
        src_mel, src_style, src_cond = feature_extract(source_wave16, source_wave22, ce, se, cfm_len_regulator, dtype)
            
        cat_condition = torch.cat([tar_cond, src_cond], dim=1)
        original_len = cat_condition.size(1)
                
        with torch.no_grad():
            with torch.autocast(device_type=cfg['device'], dtype = dtype):  # force CFM to use float32
                  # Voice Conversion
                  
                vc_mel = cfm.inference(
                    cat_condition,
                    torch.LongTensor([original_len]).to(cfg['device']),
                    tar_mel, tar_style, cfg['diff_step'],
                    inference_cfg_rate=[cfg['intelligebility_cfg_rate'], cfg['similarity_cfg_rate']],
                    random_voice=False,
                )
                vc_mel = vc_mel[:, :, tar_mel.size(2):original_len]
                vc_wave = vocoder(vc_mel).squeeze()[None]
                e = time.time()
                audlen[i,1] = e - s
                #torchaudio.save("Data/LD004C1/" + filename, vc_wave, 22050 )
            
                output =vc_wave[0].float().cpu().numpy()  
                sf.write(savedir + filename[len(filepath)+1:], output, 22050)
            
       
                
    new= {}
    new['time'] = audlen
    sio.savemat(savedir + "time_" + filename[len(filepath)+1:] + ".mat",new)
                    