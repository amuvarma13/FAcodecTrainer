import shutil
import warnings
import argparse
import torch
import os
import yaml

warnings.simplefilter('ignore')

from modules.commons import *
from hf_utils import load_custom_model_from_hf
from losses import *
import time

import torchaudio
import librosa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    print("No checkpoint path or config path provided. Loading from huggingface model hub")
    # ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
    ckpt_path = "Models/run_timbre_norm_ctc_titanet/FAcodec_epoch_00007_step_01000.pth"
    config_path = "Models/run_timbre_norm_ctc_titanet/config.yml"

    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params)

    ckpt_params = torch.load(ckpt_path, map_location="cpu")
    ckpt_params = ckpt_params['net'] if 'net' in ckpt_params else ckpt_params # adapt to format of self-trained checkpoints

    for key in ckpt_params:
        model[key].load_state_dict(ckpt_params[key])

    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    return model

def load_redecoder():

    print("No checkpoint path or config path provided. Loading from huggingface model hub")
    ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec-redecoder")


    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, stage="redecoder")

    ckpt_params = torch.load(ckpt_path, map_location="cpu")

    for key in model:
        model[key].load_state_dict(ckpt_params[key])

    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    return model

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

@torch.no_grad()
def main(args):
    model = load_model()
    redecoder = load_redecoder()
    source = args.source
    source_audio = librosa.load(source, sr=24000)[0]
    # crop only the first 30 seconds
    source_audio = source_audio[:24000 * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    # without timbre norm
    z = model.encoder(source_audio[None, ...].to(device).float())
    z, quantized, commitment_loss, codebook_loss, timbre, codes = model.quantizer(z,source_audio[None, ...].to(device).float(), n_c=2, return_codes=True)

    z1 = redecoder.encoder(codes[0], codes[1], timbre, use_p_code=False, n_c=1)
    full_pred_wave = redecoder.decoder(z1)

    full_pred_wave2 = model.decoder(z)

    os.makedirs("reconstructed", exist_ok=True)
    source_name = source.split("/")[-1].split(".")[0]
    torchaudio.save(f"reconstructed/{source_name}vmm.wav", full_pred_wave[0].cpu(), 24000)
    torchaudio.save(f"reconstructed/{source_name}vmm2.wav", full_pred_wave2[0].cpu(), 24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()
    main(args)