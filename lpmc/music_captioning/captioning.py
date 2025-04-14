import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from lpmc.utils.eval_utils import load_pretrained
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from lpmc.music_captioning.model.bart import BartCaptionModel


FILE_DIR = os.path.dirname(__file__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MSD Training')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id to use.')
    parser.add_argument('--framework', type=str, default='transfer')
    parser.add_argument('--caption_type', type=str, default='lp_music_caps')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='last')
    parser.add_argument('--audio_path', type=str, default='../../dataset/samples/orchestra.wav')
    return parser.parse_args()


def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, _ = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
        
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
        
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio


def captioning(args, store_parquet):
    save_dir = os.path.join(FILE_DIR, f"exp/{args.framework}/{args.caption_type}")
    
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    model = BartCaptionModel(max_length=config.max_length)
    model, _ = load_pretrained(args, save_dir, model, mdp=config.multiprocessing_distributed)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu).eval()

    audio_tensor = get_audio(args.audio_path)
    audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        outputs = model.generate(samples=audio_tensor, num_beams=args.num_beams)

    return process_captions(outputs, args.audio_path, store_parquet)


def process_captions(captions, audio_path, store_parquet):
    results = []
    for idx, caption in enumerate(captions):
        time_range = f"{idx * 10}:00-{(idx + 1) * 10}:00"
        entry = {
            "file": audio_path,
            "time": time_range,
            "caption": caption
        }
        results.append(entry)

    if store_parquet:
        return results
    else:
        for item in results:
            print({"time": item["time"], "text": item["caption"]})


def main(store_parquet=False):
    args = parse_arguments()
    return captioning(args, store_parquet=store_parquet)
    

if __name__ == '__main__':
    main()
