#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import shutil
import warnings
import configparser
import ast
import contextlib
import glob
import csv
from pathlib import Path
import pandas as pd
import soundfile as sf

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler
)
from pipeline.pipeline_audioldm import  AudioLDMPipeline
from transformers import SpeechT5HifiGan
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from audioldm.utils import default_audioldm_config
from scipy.io.wavfile import write
from utils.templates import imagenet_templates_small, imagenet_style_templates_small, text_editability_templates, minimal_templates, imagenet_templates_small_class
from evaluate import LAIONCLAPEvaluator

import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display
import os

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, audios=None, base_model=str, repo_folder=None):
    audio_str = ""
    for i, audio in enumerate(audios):
        write(os.path.join(repo_folder, f"audio_{i}.wav"),16000, audio)
        audio_str += f"![aud_{i}](./audio_{i}.wav)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{audio_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def log_csv(csv_file,row):
    #row is a list of strings, eg
    # row = ['Jane Smith', '28', 'Designer']

    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # File exists, open in append mode
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    else:
        # File does not exist, create and add line
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # header = ['Name', 'Age', 'Occupation']
            # writer.writerow(header)
            writer.writerow(row)

def create_mixture(waveform1, waveform2, snr):

    min_length = min(waveform1.shape[1], waveform2.shape[1])
    waveform1 = waveform1[:, :min_length]
    waveform2 = waveform2[:, :min_length]

    # Calculate the power of each waveform
    power1 = torch.mean(waveform1 ** 2)
    power2 = torch.mean(waveform2 ** 2)

    # Calculate the desired power ratio based on SNR (Signal-to-Noise Ratio)
    desired_snr = 10 ** (-snr / 10)
    scale_factor = torch.sqrt(desired_snr * power1 / power2)

    # Scale the second waveform to achieve the desired SNR
    scaled_waveform2 = waveform2 * scale_factor

    mixture = waveform1 + scaled_waveform2
    return mixture.numpy()


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step,vocoder,concept_audio_dir,placeholder_token_ids, validate_experiments=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_audio_files} audio files with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = AudioLDMPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    audios = []
    for _ in range(args.num_validation_audio_files):
        with torch.autocast("cuda"):
            audio_gen = pipeline(args.validation_prompt, num_inference_steps=50,audio_length_in_s=10, generator=generator).audios[0]
        audios.append(audio_gen)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # np_images = np.stack([np.asarray(img) for img in images])
            for i, audio in enumerate(audios):

                tracker.writer.add_audio(f"validation_{i}", audio, global_step, sample_rate=16000)
        if tracker.name == "wandb":
            # tracker.log(
            #     {
            #         "validation": [
            #             wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
            #         ]
            #     }
            # )
            raise NotImplementedError("Wandb not implemented yet for audio")
    if validate_experiments:
        # TODO check if audio length should be set, for now setting to training length
        audios_rec = pipeline(args.validation_prompt, num_inference_steps=50,audio_length_in_s=10.24, generator=generator,num_waveforms_per_prompt=10).audios
        
        # print(audios_rec.shape)
        # audios_rec = np.concatenate((audios_rec, pipeline(args.validation_prompt, num_inference_steps=50, generator=generator,num_waveforms_per_prompt=20).audios))
        # print("audios_rec shape: {}".format(audios_rec.shape))
        val_audio_dir = os.path.join(args.output_dir, "reconstruction_audio_{}".format(global_step))
        os.makedirs(val_audio_dir, exist_ok=True)
        for i, audio in enumerate(audios_rec):
            write(os.path.join(val_audio_dir, f"{'_'.join(args.validation_prompt.split(' '))}_{i}.wav"),16000, audio)
        
        print("loading clap evaluator")

        with contextlib.redirect_stdout(None):
            
            evaluator = LAIONCLAPEvaluator(device=accelerator.device)
        reconstruction_score=evaluator.audio_to_audio_similarity(concept_audio_dir, val_audio_dir)
        print("Reconstruction score: {}".format(reconstruction_score))
        print(type(reconstruction_score))
        accelerator.log({"reconstruction_score": reconstruction_score.item()},step=global_step)
        reconstruction_csv_path=os.path.join(args.output_dir, "reconstruction_score.csv")
        log_csv(reconstruction_csv_path,[global_step,reconstruction_score.item()])
        # accelerator.log({"reconstruction_scored": np.float16(reconstruction_score)},step=global_step)
        del audios_rec

        # val_audio_dir = os.path.join(args.output_dir, "editability_audio_{}".format(global_step))
        # os.makedirs(val_audio_dir, exist_ok=True)
        # for prompt in text_editability_templates:
        #     prompt=" ".join(prompt.format(("".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)))).split())
        #     audio = pipeline(prompt, num_inference_steps=50, generator=generator,num_waveforms_per_prompt=1).audios[0]
        #     write(os.path.join(val_audio_dir, f"{'_'.join(prompt.split(' '))}.wav"),16000, audio)
        
        # text_score=evaluator.text_to_audio_similarity(val_audio_dir)
        # print("Text score: {}".format(text_score))
        # accelerator.log({"text_score": text_score}, step=global_step)
        del evaluator
        
    del pipeline
    torch.cuda.empty_cache()
    return audios


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, default=None, help="Path to .ini file.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=False, help="A folder containing the training data."
    )
    parser.add_argument(
        "--file_list", type=str, default=None, help="Path to a csv file containing which files to train on from the training data directory."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=False,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument("--initializer", type=str, default="random_token",choices=["random_token","random_tokens","multitoken_word","saved_embedding","mean","multiresolution","mean_word"], help="How to initialize the placeholder.")
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=False, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--object_class", type=str, default=None, help="Choose a class to learn, works with learnable property 'object_class'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="audio-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    #todo:change resolution
    # parser.add_argument(
    #     "--resolution",
    #     type=int,
    #     default=512,
    #     help=(
    #         "The resolution for input images, all the images in the train/validation dataset will be resized to this"
    #         " resolution"
    #     ),
    # )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of audio.")
    # parser.add_argument(
    #     "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    # )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_audio_files",
        type=int,
        default=4,
        help="Number of audio files that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--validate_experiments", action="store_true", help="Whether to validate experiments.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--save_concept_audio", action="store_true",default=True, help="Whether or not to save concept audio.")
    parser.add_argument("--augment_data", action="store_true",default=False, help="Whether or not to augment the training data")
    parser.add_argument("--num_scales", type=int, default=None, help="Number of scales for multi-resolution tarining")
    parser.add_argument("--mix_data", type=str,default=None, help="If a path to an dir containing background audios is specified performs mixture training")
    parser.add_argument(
        "--snr",
        type=int,
        default=20,
        help="In mixture training specify SNR of",
    )
    parser.add_argument(
    "--num_audio_files_to_train",
    type=int,
    default=None,
    help="Number of files to use for training if None will use all files in training dir",
    )

    def read_args_from_config(filename):
        config = configparser.ConfigParser()
        config.read(filename)
        args = dict(config["Arguments"])

        # Convert the values to the appropriate data types
        for key, value in args.items():
            try:
                args[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # If the value cannot be evaluated, keep it as a string

        return args

    cli_args, _ = parser.parse_known_args()

    if cli_args.config:
        print("Reading arguments from config file")
        config_args = read_args_from_config(cli_args.config)

        # Update the argparse namespace with config_args
        for key, value in config_args.items():
            setattr(cli_args, key, value)

    args = cli_args

    # args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


from audioldm.audio.tools import get_mel_from_wav, _pad_spec, normalize_wav, pad_wav
def read_wav_file(filename, segment_length, augment_data=False):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    if augment_data:
        waveform = augment_audio(
            waveform,
            sr,
            p=0.8,
            noise=True,
            reverb=True,
            low_pass=True,
            pitch_shift=True,
            delay=True)
        
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)
    
    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform


def wav_to_fbank(
        filename,
        target_length=1024,
        fn_STFT=None,
        augment_data=False,
        mix_data=False,
        snr=None
    ):
    assert fn_STFT is not None

    # mixup
    if mix_data:
        assert snr is not None, "You specified mixed training but didn't provide SNR!"
        background_file_paths = [os.path.join(mix_data, p) for p in os.listdir(mix_data)]
        background_file_path = random.sample(background_file_paths,1)[0]
        waveform = read_wav_file(filename, target_length * 160, augment_data=augment_data)
        background = read_wav_file(background_file_path, target_length * 160)
        waveform = create_mixture(torch.tensor(waveform), torch.tensor(background), snr)
    else:
        waveform = read_wav_file(filename, target_length * 160, augment_data=augment_data)  # hop size is 160

    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform



def wav_to_mel(
        original_audio_file_path,
        duration,
        augment_data=False,
        mix_data=False,
        snr=None
):
    config=default_audioldm_config()
    
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path,
        target_length=int(duration * 102.4),
        fn_STFT=fn_STFT,
        augment_data=augment_data,
        mix_data=mix_data,
        snr=snr
    )
    mel = mel.unsqueeze(0)
    # mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    if augment_data:
        mel=mel.unsqueeze(0)
        mel = augment_spectrogram(mel)
        mel = mel.squeeze(0)
    return mel


class AudioInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        text_encoder,
        device,
        audioldmpipeline,
        learnable_property="object",  # [object, style, minimal]
        sample_rate=16000,
        duration=2.0,
        repeats=100,
        set="train",
        placeholder_token="*",
        file_list=None,
        object_class=None,
        augment_data=False,
        multiresolution=False,
        train_timesteps=None,
        num_scales=None,
        mix_data=False,
        snr=None,
        num_files_to_train=None
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.sample_rate = sample_rate
        self.duration = duration
        self.placeholder_token = placeholder_token
        self.audioldmpipeline = audioldmpipeline
        self.text_encoder = text_encoder
        self.num_files_to_train = num_files_to_train

        if file_list is not None:
            file_list=list(pd.read_csv(file_list, header=None)[0])
            self.audio_files = [os.path.join(self.data_root, file_path) for file_path in file_list]
        else:
            self.audio_files = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if file_path.endswith(".wav")]
        if self.num_files_to_train:
            self.audio_files = sorted(self.audio_files)[:self.num_files_to_train]
        self.num_files = len(self.audio_files)
        self._length = self.num_files
        self.device = device
        self.augment_data = augment_data
        self.multiresolution = multiresolution
        self.num_train_timesteps = train_timesteps
        self.num_scales = num_scales
        self.mix_data = mix_data
        self.snr = snr

        if self.learnable_property == "object":
            self.templates = imagenet_templates_small
        elif self.learnable_property == "style":
            self.templates = imagenet_style_templates_small
        elif self.learnable_property == "minimal":
            self.templates = minimal_templates
        elif self.learnable_property == "object_class":
            self.templates = imagenet_templates_small_class
            self.object_class = object_class
        if set == "train":
            self._length = self.num_files * repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        audio_file = self.audio_files[i % self.num_files]
        example["mel"]=wav_to_mel(
            audio_file,
            self.duration,
            augment_data=self.augment_data,
            mix_data=self.mix_data,
            snr=self.snr
)


        # Sample a random timestep for diffusion
        timestep = torch.randint(
            0, self.num_train_timesteps, (1,)
        ).long()

        if self.multiresolution:
            
            scale_index = int((timestep / self.num_train_timesteps) * self.num_scales)
            placeholder_string = f"<{self.placeholder_token.split('|')[0][1:]}|{scale_index}|>"
            text = random.choice(self.templates).format(placeholder_string)
        else:
            placeholder_string = self.placeholder_token
            if self.learnable_property == "object_class":
                text = random.choice(self.templates).format(placeholder_string, self.object_class)
            else:
                text = random.choice(self.templates).format(placeholder_string)
        
        # text_inputs = self.tokenizer(
        #         text,
        #         padding="max_length",
        #         max_length=self.tokenizer.model_max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        # text_input_ids = text_inputs.input_ids
        # attention_mask = text_inputs.attention_mask
        # prompt_embeds = self.text_encoder(
        #         text_input_ids.to(self.device),
        #         attention_mask=attention_mask.to(self.device),
        #     )
        # prompt_embeds = prompt_embeds.text_embeds
        # prompt_embeds = F.normalize(prompt_embeds, dim=-1)
        # prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
        # example["prompt_embeds"] = prompt_embeds
        
        #alternatively, we can get prompt embeds by the following from the audioldmpipeline
        #there we can also do classifier free guidance during training if we want

        example["prompt_embeds"] = self.audioldmpipeline._encode_prompt(

            prompt=text,
            device=self.device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=False
        )
        return example, timestep



def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = ClapTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    vocoder = SpeechT5HifiGan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vocoder", revision=args.revision
    )

    audioldmpipeline=AudioLDMPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        vocoder=vocoder,
        scheduler=noise_scheduler,
        tokenizer=tokenizer
    ).to(accelerator.device)
    
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]
    
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")
    
    init_embeddings_save_path=""
    if args.initializer=="mean_word":
        print("Starting from the mean embedding of the placeholder token subwords.")

        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        if len(token_ids) > 1:
            token_embeds = text_encoder.get_input_embeddings().weight.data
            init_token_embed=token_embeds[token_ids].mean(dim=0)


        if args.num_vectors > 1:
            print("Adding {} tokens to the tokenizer.".format(str(args.num_vectors)))

            # add dummy tokens for multi-vector
            additional_tokens = []
            for i in range(1, args.num_vectors):
                additional_tokens.append(f"{args.placeholder_token}_{i}")
            placeholder_tokens += additional_tokens
        
        # the following is wrong
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # initializer_token_id = token_ids[0]
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("placeholder_token_ids: ", placeholder_token_ids)


        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with a random embedding, and save the embedding to file
        token_embeds = text_encoder.get_input_embeddings().weight.data

        with torch.no_grad():
            # random_embedding = torch.rand(768).to(token_embeds.device)
            #save embedding to file
            init_embed_dict = {}
            init_embeddings_save_path=os.path.join(args.output_dir, "initial_embeds.bin")
           
            for token_id in placeholder_token_ids:
                init_embed_dict [tokenizer.convert_ids_to_tokens(token_id)] = init_token_embed.clone().detach().cpu().unsqueeze(0)

                # token_embeds[token_id] = token_embeds[initializer_token_id].clone()
                token_embeds[token_id] = init_token_embed.clone()
            if not args.resume_from_checkpoint:
                torch.save(init_embed_dict, init_embeddings_save_path)

    elif args.initializer=="random_token":
        print("Starting from a random embedding. If num_tokens>1, all tokens start with the same embedding.")

        if args.num_vectors > 1:
            print("Adding {} tokens to the tokenizer.".format(str(args.num_vectors)))

            # add dummy tokens for multi-vector
            additional_tokens = []
            for i in range(1, args.num_vectors):
                additional_tokens.append(f"{args.placeholder_token}_{i}")
            placeholder_tokens += additional_tokens
        
        # the following is wrong
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # initializer_token_id = token_ids[0]
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("placeholder_token_ids: ", placeholder_token_ids)


        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with a random embedding, and save the embedding to file
        token_embeds = text_encoder.get_input_embeddings().weight.data

        with torch.no_grad():
            random_embedding = torch.rand(768).to(token_embeds.device)
            #save embedding to file
            init_embed_dict = {}
            init_embeddings_save_path=os.path.join(args.output_dir, "initial_embeds.bin")
           
            for token_id in placeholder_token_ids:
                init_embed_dict [tokenizer.convert_ids_to_tokens(token_id)] = random_embedding.clone().detach().cpu().unsqueeze(0)

                # token_embeds[token_id] = token_embeds[initializer_token_id].clone()
                token_embeds[token_id] = random_embedding.clone()
            if not args.resume_from_checkpoint:
                torch.save(init_embed_dict, init_embeddings_save_path)

    elif args.initializer=="multiresolution":

        assert args.num_vectors == 1, "Multiresolution training works with 1 vector per resolution for now"
        placeholder_tokens = []
        for scale_index in range(args.num_scales):
            placeholder_string = f"<{args.placeholder_token}|{scale_index}|>"
            placeholder_tokens.append(placeholder_string)
        increment = tokenizer.add_tokens(placeholder_tokens)
        print("incriment:", increment)
            # if increment == 0:
            #     raise ValueError(
            #         f"The tokenizer already contains the token {placeholder_string}. Please pass a different"
            #         " `placeholder_token` that is not already in the tokenizer."
            #     )

        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("placeholder_token_ids: ", placeholder_token_ids)


        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with a random embedding, and save the embedding to file
        token_embeds = text_encoder.get_input_embeddings().weight.data

        with torch.no_grad():
            random_embedding = torch.rand(768).to(token_embeds.device)
            #save embedding to file
            init_embed_dict = {}
            init_embeddings_save_path=os.path.join(args.output_dir, "initial_embeds.bin")
           
            for token_id in placeholder_token_ids:
                init_embed_dict [tokenizer.convert_ids_to_tokens(token_id)] = random_embedding.clone().detach().cpu().unsqueeze(0)

                # token_embeds[token_id] = token_embeds[initializer_token_id].clone()
                token_embeds[token_id] = random_embedding.clone()
            if not args.resume_from_checkpoint:
                torch.save(init_embed_dict, init_embeddings_save_path)


    elif args.initializer=="random_tokens":
        print("Starting from a random embedding. Making a random embedding for each placeholder token.")

        if args.num_vectors > 1:
            print("Adding {} tokens to the tokenizer.".format(str(args.num_vectors)))

            # add dummy tokens for multi-vector
            additional_tokens = []
            for i in range(1, args.num_vectors):
                additional_tokens.append(f"{args.placeholder_token}_{i}")
            placeholder_tokens += additional_tokens
        
        # the following is wrong
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("placeholder_token_ids: ", placeholder_token_ids)


        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with a random embedding, and save the embedding to file
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            init_embed_dict = {}
            init_embeddings_save_path=os.path.join(args.output_dir, "initial_embeds.bin")
            for token_id in placeholder_token_ids:
                random_embedding = torch.rand(768).to(token_embeds.device)
                init_embed_dict[tokenizer.convert_ids_to_tokens(token_id) ] = random_embedding.detach().cpu()
                token_embeds[token_id] = random_embedding.clone()
            if not args.resume_from_checkpoint:
                torch.save(init_embed_dict, init_embeddings_save_path)
    elif args.initializer=="multitoken_word":
        
        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)

        print("Initializer word: ", args.initializer_token)
        print("Initializer token ids: ",token_ids)
        # RobertaTokenizerFast.encode returns a list of ids, so we add additional tokens as needed
        if len(token_ids) > 1:
            args.num_vectors += len(token_ids) - 1
        else:
            print("You selected initializer=multitoken_word, but the initializer_token is a single token.")
            exit()

        # add dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, args.num_vectors):
            additional_tokens.append(f"{args.placeholder_token}_{i}")
        placeholder_tokens += additional_tokens

        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        initializer_token_ids = token_ids
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            init_embed_dict = {}
            init_embeddings_save_path=os.path.join(args.output_dir, "initial_embeds.bin")
            for token_id,initializer_token_id in zip(placeholder_token_ids,initializer_token_ids):
                init_embed_dict[tokenizer.convert_ids_to_tokens(token_id) ] = token_embeds[initializer_token_id].clone().detach().cpu()
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            if not args.resume_from_checkpoint:
                torch.save(init_embed_dict, init_embeddings_save_path)

    



    elif args.initializer=="saved_embedding":
        print("Starting from a saved embedding or embeddings of a concept.")
        print("Will disregard num_vectors, placeholder token arguments.")

        try:
            embedding_dict=torch.load(args.initializer_token)
            main_token=list(embedding_dict.keys())[0]
            n_tokens=len(embedding_dict[main_token])
            # n_tokens=1
            main_token=args.placeholder_token
            placeholder_tokens=[main_token]
            placeholder_tokens+=[main_token+"_"+str(i) for i in range(1,n_tokens)]
            embeddings = list(embedding_dict.values())[0]
        except:
            raise ValueError("Could not load embedding from file. Please check the path.")
        
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with a random embedding, and save the embedding to file
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id,embedding in zip(placeholder_token_ids,embeddings):
                embedding.to(token_embeds.device)
                token_embeds[token_id] = embedding.clone()
    elif args.initializer=="mean":
        print("Starting from the mean vector embedding.")
        token_embeds = text_encoder.get_input_embeddings().weight.data
        mean_embed=torch.mean(token_embeds, dim=0)
        embeddings=[mean_embed for i in range(args.num_vectors)]
        placeholder_tokens=[args.placeholder_token]
        placeholder_tokens+=[args.placeholder_token+"_"+str(i) for i in range(args.num_vectors-1)]
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with a random embedding, and save the embedding to file
        token_embeds = text_encoder.get_input_embeddings().weight.data
        init_embeddings_save_path=os.path.join(args.output_dir, "initial_embeds.bin")
        init_embed_dict = {}
        with torch.no_grad():
            for token_id,embedding in zip(placeholder_token_ids,embeddings):
                init_embed_dict[tokenizer.convert_ids_to_tokens(token_id) ] = embedding.clone().detach().cpu()
                embedding.to(token_embeds.device)
                token_embeds[token_id] = embedding.clone()
            if not args.resume_from_checkpoint:
                torch.save(init_embed_dict, init_embeddings_save_path)
    
    print("placeholder_tokens: ", tokenizer.convert_ids_to_tokens(placeholder_token_ids))
    args.validation_prompt=args.validation_prompt.replace(args.placeholder_token, ("".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))))
    print("validation_prompt: ", args.validation_prompt)
          
    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    # text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embeddings.requires_grad_(False)
    text_encoder.text_model.embeddings.token_type_embeddings.requires_grad_(False)
    text_encoder.text_model.embeddings.LayerNorm.requires_grad_(False)
    text_encoder.text_model.pooler.requires_grad_(False)
    text_encoder.text_projection.requires_grad_(False)

    for name, param in text_encoder.named_parameters():
        if param.requires_grad:
            print(name)
    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # save concept audio files to directory
    if args.save_concept_audio:
        
        concept_audio_dir = os.path.join(args.output_dir, "training_audio")
        os.makedirs(concept_audio_dir, exist_ok=True)
        if args.file_list:
            file_list=list(pd.read_csv(args.file_list, header=None)[0])
            audio_files = [os.path.join(args.train_data_dir, file_path) for file_path in file_list]
        else:
             audio_files = [
                os.path.join(args.train_data_dir, file_path) for file_path in os.listdir(args.train_data_dir) if file_path.endswith(".wav")
            ]
        if args.num_audio_files_to_train:
            audio_files = sorted(audio_files)[:args.num_audio_files_to_train]
        
        for audio_file in audio_files:
            wave,sr=librosa.load(audio_file, sr=args.sample_rate)
            # wave=wave[:args.duration*sr]
            save_path=os.path.join(concept_audio_dir, os.path.basename(audio_file))
            sf.write(save_path, wave, sr)

            # shutil.copy(audio_file, concept_audio_dir)
        
    else:
        if args.file_list:
            # if file list is provided, we assume that the concept audio files are in the same directory as the file list
            concept_audio_dir = None
        else:
            concept_audio_dir = args.train_data_dir

    
    # Dataset and DataLoaders creation:
    train_dataset = AudioInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        sample_rate=args.sample_rate,
        duration=args.duration,
        placeholder_token=("".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        set="train",
        device=accelerator.device,
        audioldmpipeline=audioldmpipeline,
        file_list=args.file_list,
        object_class=args.object_class,
        augment_data=True if args.augment_data else False,
        multiresolution=True if args.initializer=="multiresolution" else False,
        train_timesteps=noise_scheduler.config.num_train_timesteps,
        num_scales=args.num_scales,
        num_files_to_train=args.num_audio_files_to_train
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))
    accelerator.log({"Placeholder token": args.placeholder_token})
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            # if we're not going to resume from checkpoint, we need to save the initial embeddings
            if init_embeddings_save_path:
                print("Saving initial embeddings")
                torch.save(init_embed_dict, os.path.join(args.output_dir, "initial_embeds.bin"))
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            #we load the initial embedding, so we can compare it to the current embedding
            init_embeds_path=os.path.join(args.output_dir, "initial_embeds.bin")
            
            if os.path.isfile(init_embeds_path):
                init_embeds=torch.load(init_embeds_path)
                init_embeds=torch.stack(list(init_embeds.values())).to(accelerator.device)

            # orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    # print("orig_embeds_params: ", orig_embeds_params.shape)
    # print("placeholder_token_ids: ", placeholder_token_ids)
    # print(orig_embeds_params[min(placeholder_token_ids) : max(placeholder_token_ids) + 1].shape)
    if not args.resume_from_checkpoint:
        init_embeds=orig_embeds_params[min(placeholder_token_ids) : max(placeholder_token_ids) + 1].clone()
    # init_embeds_path=os.path.join(args.output_dir, "initial_embeds.bin")
    # init_embeds=torch.load(init_embeds_path)
    # init_embeds=torch.stack(list(init_embeds.values()))
    print("init_embeds: ", init_embeds.shape)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, (batch, timesteps) in enumerate(train_dataloader):
            # print("step: ", step)
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            # print("mel: ", batch["mel"].shape)

            # save_path = 'mel_test.jpg'
            # mel=batch["mel"].squeeze().clone().detach().cpu().numpy()[:,:699].T
            # librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000,hop_length=160,fmax=9000)
            # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
            # pylab.close()

            # reconstructed_audio = vocoder(batch["mel"].to("cpu",dtype=weight_dtype).squeeze(1))
            # # print("reconstructed_audio: ", reconstructed_audio.shape)
            # write("reconstructed_audio.wav",16000,reconstructed_audio[0].detach().cpu().numpy())
            
            with accelerator.accumulate(text_encoder):

                # Convert audios to latent space
                # latents = vae.encode(batch["mel"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = vae.encode(batch["mel"].to(dtype=weight_dtype)).latent_dist.sample()
                
                # print("latent_dim: ", latents.shape)

                # save_path = 'latent_plot.jpg'
                # mel=latents.squeeze().clone().detach().cpu().numpy()[0,:,:].T
                # print(mel.shape)
                # librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000,hop_length=160,fmax=9000)
                # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
                # pylab.close()

                # decoded = vae.decode(latents).sample

                # save_path = 'mel_vae_mel.jpg'
                # mel=decoded.squeeze().clone().detach().cpu().numpy()[:,:699].T
                # librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000,hop_length=160,fmax=9000)
                # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
                # pylab.close()

                # reconstructed_latents = vocoder(decoded.squeeze(1).detach().cpu())
                # write("mel_vae_mel_vocoder.wav",16000,reconstructed_latents[0].detach().cpu().numpy())

                
                latents = latents * vae.config.scaling_factor
                # print("latents: ", latents.shape)
                

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # # Sample a random timestep for each image
                timesteps = timesteps.squeeze(0).to(latents.device)

                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                # timesteps = timesteps.long()
                # print("time_steps: ", timesteps)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # save_path = 'latent_plot_noisy.jpg'
                # mel=noisy_latents.squeeze().clone().detach().cpu().numpy()[0,:,:].T
                # print(mel.shape)
                # librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000,hop_length=160,fmax=9000)
                # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
                # pylab.close()

                # decoded = vae.decode(noisy_latents).sample

                # save_path = 'mel_after.jpg'
                # mel=decoded.squeeze().clone().detach().cpu().numpy()[:,:699].T
                # librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000,hop_length=160,fmax=9000)
                # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
                # pylab.close()

                
                # print("decoded_mel: ", decoded.shape)
                
                # reconstructed_latents = vocoder(decoded.squeeze(1).detach().cpu())
                # write("mel_vae_noise_mel_vocoder.wav",16000,reconstructed_latents[0].detach().cpu().numpy())
                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                prompt_embeds = batch["prompt_embeds"]
                
                prompt_embeds = prompt_embeds.squeeze(-2)

                # Predict the noise residual

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=None,class_labels=prompt_embeds).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                # print("min_placeholder_token_ids: ", min(placeholder_token_ids))
                # print("max_placeholder_token_ids: ", max(placeholder_token_ids))
                # print("index_no_updates are zero in positions: ", torch.where(index_no_updates==0))
                
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                audios = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        audios = log_validation(
                            text_encoder, 
                            tokenizer, 
                            unet, vae, args, accelerator, weight_dtype, global_step, vocoder,
                            concept_audio_dir, 
                            placeholder_token_ids=placeholder_token_ids, 
                            validate_experiments=args.validate_experiments
                        )
            current_emb = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
            cosine_sim_init = F.cosine_similarity(init_embeds, current_emb)
            embeddingnorm = current_emb.norm(2,dim=1)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], }
            for i,token in enumerate(placeholder_tokens):
                logs[token+"_cosine_sim"] = cosine_sim_init[i].detach().item()
                logs[token+"_embedding_norm"] = embeddingnorm[i].detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                audios=audios,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
