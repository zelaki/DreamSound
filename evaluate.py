import torch
import json
# from clap_model.ase_model import ASE
from ruamel import yaml
import sys
import librosa
import torch.nn.functional as F
import numpy as np
import os
import argparse
import re
import csv
import laion_clap
from scipy.io.wavfile import write
from utils import templates
from pipeline.pipeline_audioldm import AudioLDMPipeline
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
from frechet_audio_distance import FrechetAudioDistance
import pandas as pd
from accelerate.utils import set_seed
import shutil
def write_to_csv(path, score, t):
    """
    exp_name: Name of experiment. Could be <oud>
    score: CLAP_A or CLAP_T score
    t: type of score 
    """
    row = [score, t]
    # 'a' mean append. We will append to the same csv the new results
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(
        description="An evaluation of audio textual inversion and dreambooth using CLAP_A and CLAP_T and FAD scores"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="The superdir with the experiments",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="results.csv",
        help="Path a csv file to save the results",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dreambooth",
        help="Path a csv file to save the results",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--step",
        default="last",
        help="Step to use for evaluation",
    )
    parser.add_argument(
        "--clap_ckpt",
        default="music_audioset_epoch_15_esc_90.14.pt",
        help="CLAP model ckpt to use for audio/text similarity",
    )


    args = parser.parse_args()
    return args



class CLAPEvaluator(object):

    def __init__(
            self,
            device,
            clap_config='inference.yaml',
            clap_param_path="/home/theokouz/data/WavCaps/cnn14-bert.pt"
        ) -> None:
        """
        You can download the CLAP model parameteres from:
        https://drive.google.com/drive/folders/1MeTBren6LaLWiZI8_phZvHvzz4r9QeCD
        Download the model named CNN14-BERT-PT.pt and set <clap_param_path> to your local path.
        """
        self.device = device

        with open(clap_config, "r") as f:
            config = yaml.safe_load(f)
        self.model = ASE(config)
        self.model.to(device)
        cp = torch.load(clap_param_path)
        self.model.load_state_dict(cp['model'])
        self.model.eval()


    def prepare_text(self, generated_audio_dir: str):
        """
        This method assumes the <generated_audio_dir>
        contains wavs with names:
        - a_recording_of_an_<oud>_0.wav
        - ...
        - a_recording_of_an_<oud>_63.wav

        Return the prompting text with indexes and placeholder token removed.
        In the above example the method will return "a recording of an".
        """
        generated_audio_paths = [os.path.join(generated_audio_dir, p) for p in os.listdir(generated_audio_dir)]
        prompt_used_for_generation = " ".join(os.path.basename(generated_audio_paths[0]).split("_")[:-1])
        return re.sub(r"<.*?>", "", prompt_used_for_generation)


    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        return self.model.encode_text([text])

    @torch.no_grad()
    def encode_audio(self, audio_paths: str) -> torch.Tensor:
        audios = []
        for audio_path in audio_paths:
            audio, _ = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
            if audio.shape[-1] < 32000 * 10:
                pad_length = 32000 * 10 - audio.shape[-1]
                audio = F.pad(audio, [0, pad_length], "constant", 0.0)
            audios.append(audio)
        audios_tensor = torch.cat(audios)
        return self.model.encode_audio(audios_tensor)
    
    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        text_features = self.encode_text(text).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_audio_features(self, audio_paths: list, norm: bool = True) -> torch.Tensor:
        audio_features = self.encode_audio(audio_paths)
        
        if norm:
            audio_features /= audio_features.clone().norm(dim=-1, keepdim=True)

        return audio_features

    def audio_to_audio_similarity(self, src_audio_dir, generated_audio_dir):
        src_audio_paths = [os.path.join(src_audio_dir, p) for p in os.listdir(src_audio_dir)]
        generated_audio_paths = [os.path.join(generated_audio_dir, p) for p in os.listdir(generated_audio_dir)]

        src_audio_features = self.get_audio_features(src_audio_paths)
        gen_audio_features = self.get_audio_features(generated_audio_paths)

        return (src_audio_features @ gen_audio_features.T).mean()

    def txt_to_audio_similarity(self, generated_audio_dir):
        text = self.prepare_text(generated_audio_dir)
        generated_audio_paths = [os.path.join(generated_audio_dir, p) for p in os.listdir(generated_audio_dir)]
        text_features = self.get_text_features(text)
        gen_audio_features = self.get_audio_features(generated_audio_paths)

        return (text_features @ gen_audio_features.T).mean()


class LAIONCLAPEvaluator(object):

    def __init__(
            self,
            device,
            use_laion_clap=True,
            laion_clap_fusion=False,
            laion_clap_checkpoint='music_speech_audioset_epoch_15_esc_89.98.pt',
            clap_config='inference.yaml',
            clap_param_path="/home/theokouz/data/WavCaps/cnn14-bert.pt",
        ) -> None:
        """
        You can download the CLAP model parameteres from:
        https://drive.google.com/drive/folders/1MeTBren6LaLWiZI8_phZvHvzz4r9QeCD
        Download the model named CNN14-BERT-PT.pt and set <clap_param_path> to your local path.
        """
        self.use_laion_clap = use_laion_clap
        self.laion_clap_fusion = laion_clap_fusion
        if self.use_laion_clap:
            self.device = device
            # device = torch.device('cuda:0')
            if laion_clap_fusion:

                self.model = laion_clap.CLAP_Module(enable_fusion=True, device=self.device)
                self.model.load_ckpt(laion_clap_checkpoint,verbose=False) # download the default pretrained checkpoint.
            else:
                self.model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device, amodel= 'HTSAT-base')
                self.model.load_ckpt(laion_clap_checkpoint,verbose=False) # download the default pretrained checkpoint.
        else:   
            with open(clap_config, "r") as f:
                config = yaml.safe_load(f)
            self.model = ASE(config)
            self.model.to(device)
            cp = torch.load(clap_param_path)
            self.model.load_state_dict(cp['model'])
            self.model.eval()

    import json
    def prepare_text(self, generated_audio_dir: str):
        """
        This method assumes the <generated_audio_dir>
        contains wavs with names:
        - a_recording_of_an_<oud>_0.wav
        - ...
        - a_recording_of_an_<oud>_63.wav

        Return the prompting text with indexes and placeholder token removed.
        In the above example the method will return "a recording of an".
        """
        exp_dir_path = os.path.dirname(os.path.dirname(generated_audio_dir))
        concept_name = os.path.basename(exp_dir_path)

        with open(os.path.join("dataset/concepts/", concept_name, "class_name.txt")) as fd:
            object_class = [ln.rstrip() for ln in fd.readlines()]
            object_class = object_class[0]
        generated_audio_paths = [os.path.join(generated_audio_dir, p) for p in os.listdir(generated_audio_dir)]
        prompts_used_for_generation = []
        for path in generated_audio_paths:
            prompt_used_for_generation = " ".join(os.path.basename(path).split("_")[:-1])
            prompt_used_for_generation=re.sub(r"<.*?>", object_class, prompt_used_for_generation)
            prompt_used_for_generation = " ".join(prompt_used_for_generation.split())
            prompts_used_for_generation.append(prompt_used_for_generation)
            print("prompt_used_for_generation:", prompt_used_for_generation)

        return prompts_used_for_generation

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        if self.use_laion_clap:
           
            return self.model.get_text_embedding(text,use_tensor=True)[0]

    @torch.no_grad()
    def encode_audio(self, audio_paths: str) -> torch.Tensor:
        audios = []
        for audio_path in audio_paths:
            audio, _ = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
            if audio.shape[-1] < 32000 * 10:
                pad_length = 32000 * 10 - audio.shape[-1]
                audio = F.pad(audio, [0, pad_length], "constant", 0.0)
            audios.append(audio)
        audios_tensor = torch.cat(audios)
        return self.model.encode_audio(audios_tensor)
    
    def get_text_features(self, text: list, norm: bool = True) -> torch.Tensor:

        text_features = self.encode_text(text).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def encode_audio_batched(self,audio_paths: list, batch_size: int = 10) -> torch.Tensor:
        if batch_size is None or batch_size > len(audio_paths) or batch_size < 1:
            batch_size = len(audio_paths)
        batches= [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]
        audio_features = []
        for batch in batches:
            embeddings=self.model.get_audio_embedding_from_filelist(batch,use_tensor=True).detach()
            audio_features.append(embeddings)
        return torch.cat(audio_features)
        

    def get_audio_features(self, audio_paths: list, norm: bool = True, batch_size: int = 10) -> torch.Tensor:
        if self.use_laion_clap:
            audio_features = self.encode_audio_batched(audio_paths, batch_size=batch_size)
        else:
            audio_features = self.encode_audio(audio_paths)
            
        if norm:
            audio_features /= audio_features.clone().norm(dim=-1, keepdim=True)

        return audio_features

    def audio_to_audio_similarity(self, src_audio_dir, generated_audio_dir):
        src_audio_paths = [os.path.join(src_audio_dir, p) for p in os.listdir(src_audio_dir) if p.endswith(".wav")]
        generated_audio_paths = [os.path.join(generated_audio_dir, p) for p in os.listdir(generated_audio_dir) if p.endswith(".wav")]
        
        src_audio_features = self.get_audio_features(src_audio_paths)
        gen_audio_features = self.get_audio_features(generated_audio_paths)

        return (src_audio_features @ gen_audio_features.T).mean()

    def text_to_audio_similarity(self, generated_audio_dir):
        text = self.prepare_text(generated_audio_dir)
        print("Yoooooooooo",text)
        generated_audio_paths = [os.path.join(generated_audio_dir, p) for p in os.listdir(generated_audio_dir) if p.endswith(".wav")]
        text_features = self.get_text_features(text)
        gen_audio_features = self.get_audio_features(generated_audio_paths)

        return (text_features @ gen_audio_features.T).mean()
    
    def inter_audio_similarity(self,audio_dir):
        audio_paths = [os.path.join(audio_dir, p) for p in os.listdir(audio_dir) if p.endswith(".wav")]
        similarities=[]
        for i in range(len(audio_paths)):
            audio_paths_minus_one=audio_paths[:i]+audio_paths[i+1:len(audio_paths)]
            audio_path=[audio_paths[i]]
            audio_features_many = self.get_audio_features(audio_paths_minus_one)
            audio_features_one = self.get_audio_features(audio_path)
            similarity= (audio_features_one @ audio_features_many.T).mean()
            similarities.append(similarity.cpu().numpy())
        return np.mean(similarities)
    

class ExperimentEvaluator(object):

    def __init__(
        self,
        device,
        clap_evaluator,
        method="tinv", # tinv or dreambooth
        audioldm_model_path="audioldm-m-full",
        # audioldm_model_path="audioldm2-music",
        use_audioldm2=False
    ):
        self.device = device
        self.clap_evaluator = clap_evaluator
        self.audioldm_model_path = audioldm_model_path
        self.use_audioldm2=use_audioldm2

        
    def create_experiment_audio_tinv(self, path_to_embedding,experiment_prompts, n_audio_files_per_prompt=10, 
                                     experiment_type="reconstruction",
                                     delete_old_files=True,
                                     random_seed=None):
        experiment_audio_dir=os.path.join(os.path.dirname(path_to_embedding),f"{experiment_type}_audio")
        audioldmpipeline=AudioLDMPipeline.from_pretrained(self.audioldm_model_path).to("cuda")
        audioldmpipeline.load_textual_inversion(path_to_embedding)
        generator = None if random_seed is None else torch.Generator(device=audioldmpipeline.device).manual_seed(random_seed)
        # if delete_old_files:
        #     os.system(f"rm -rf {experiment_audio_dir}")
        
        embeddings_dict=torch.load(path_to_embedding)
        base_token=list(embeddings_dict.keys())[0]
        tokens=[base_token]
        embeds=embeddings_dict[base_token]
        if len(embeds)>1:
            for i in range(1,len(embeds)):
                tokens.append(base_token+"_"+str(i))
        token="".join(tokens)

        os.makedirs(experiment_audio_dir,exist_ok=True)
        for prompt in experiment_prompts:
            prompt=prompt.format(token)
            print(prompt)
            
            audio_files=audioldmpipeline(prompt,num_inference_steps=50,num_waveforms_per_prompt=n_audio_files_per_prompt,audio_length_in_s=10.0,generator=generator).audios
            
            for i,w in enumerate(audio_files):
                audio_name="_".join(prompt.split(" "))+"_"+str(i)+".wav"
                if experiment_type=="editability":
                    # save audio files in subfolders
                    os.makedirs(os.path.join(experiment_audio_dir,prompt),exist_ok=True)
                    save_path=os.path.join(experiment_audio_dir,prompt,audio_name)
                else:
                    save_path=os.path.join(experiment_audio_dir,audio_name)

                write(save_path, 16000, w)

        return experiment_audio_dir
    
    def create_experiment_audio_dreambooth(
            self,
            path_to_pipeline,
            experiment_prompts,
            n_audio_files_per_prompt=10,
            experiment_type="reconstruction",
                                     delete_old_files=True,
                                     random_seed=None):
        experiment_audio_dir=os.path.join(os.path.dirname(path_to_pipeline),f"{experiment_type}_audio")
        if self.use_audioldm2:
            audioldmpipeline=AudioLDM2Pipeline.from_pretrained(path_to_pipeline,use_safetensors=True).to("cuda")
        else:
            audioldmpipeline=AudioLDMPipeline.from_pretrained(path_to_pipeline,use_safetensors=True).to("cuda")
        generator = None if random_seed is None else torch.Generator(device=audioldmpipeline.device).manual_seed(random_seed)

        with open(os.path.join(os.path.dirname(path_to_pipeline),"class_name.json"), "r") as f:
            class_words = json.load(f)
            if "object_class" not in class_words.keys() or "instance_word" not in class_words.keys():
                object_class=""
                instance_word=""
            else:
                object_class=class_words["object_class"]
                instance_word=class_words["instance_word"]
        token=instance_word+" "+object_class


        os.makedirs(experiment_audio_dir,exist_ok=True)
        for prompt in experiment_prompts:
            prompt_to_gen=prompt.format(token)
            # saving a prompt with brackets so that we know the extra words
            prompt_to_save=prompt.format(f"<{token}>")
            
            audio_files=audioldmpipeline(prompt_to_gen,num_inference_steps=50,num_waveforms_per_prompt=n_audio_files_per_prompt,audio_length_in_s=10.0,generator=generator).audios
            
            for i,w in enumerate(audio_files):
                audio_name="_".join(prompt_to_save.split(" "))+"_"+str(i)+".wav"
                print("audio_name",audio_name)
                if experiment_type=="editability":
                    # save audio files in subfolders
                    os.makedirs(os.path.join(experiment_audio_dir,prompt_to_save),exist_ok=True)
                    save_path=os.path.join(experiment_audio_dir,prompt_to_save,audio_name)
                else:
                    save_path=os.path.join(experiment_audio_dir,audio_name)

                write(save_path, 16000, w)


        return experiment_audio_dir
    
    
    def reconstruction_score_tinv(self,path_to_embedding,
                                  source_dir="",
                                  reconstruction_dir="",
                                  reconstruction_prompts=["a recording of a {}"], 
                                  n_audio_files_per_prompt=4,
                                  create_audio=True,
                                  random_seed=None
                                  ):
        if not source_dir:
            source_dir=os.path.join(os.path.dirname(path_to_embedding),"training_audio")
        if create_audio:
            reconstruction_dir=self.create_experiment_audio_tinv(path_to_embedding,reconstruction_prompts, n_audio_files_per_prompt=n_audio_files_per_prompt, experiment_type="reconstruction")
        else:
            reconstruction_dir=os.path.join(os.path.dirname(path_to_embedding),"reconstruction_audio")
        print("Source dir: ", source_dir)
        print("Reconstruction dir: ", reconstruction_dir)
        reconstruction_score=self.clap_evaluator.audio_to_audio_similarity(source_dir,reconstruction_dir)
        print("Reconstruction score: ", reconstruction_score.item())
        
        frechet_vgg = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        frechet_vgg_score=frechet_vgg.score(source_dir,reconstruction_dir,dtype="float32")
        print("Frechet score VGGish: ", frechet_vgg_score.item())
        frechet_pann = FrechetAudioDistance(
            model_name="pann",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        frechet_pann_score=frechet_pann.score(source_dir,reconstruction_dir,dtype="float32")
        print("Frechet score PANN: ", frechet_pann_score.item())
        return reconstruction_score.item(), frechet_vgg_score.item(), frechet_pann_score.item()

    def reconstruction_score_dreambooth(self,path_to_pipeline,
                                  source_dir="",
                                  reconstruction_dir="",
                                  reconstruction_prompts=["a recording of a {}"], 
                                  n_audio_files_per_prompt=10,
                                  create_audio=True,
                                  random_seed=None
                                  ):
        if not source_dir:
            source_dir=os.path.join(os.path.dirname(path_to_pipeline),"training_audio")
        if create_audio:
            reconstruction_dir=self.create_experiment_audio_dreambooth(path_to_pipeline,reconstruction_prompts, n_audio_files_per_prompt=n_audio_files_per_prompt, experiment_type="reconstruction")
        else:
            reconstruction_dir=os.path.join(os.path.dirname(path_to_pipeline),"reconstruction_audio")
        print("Source dir: ", source_dir)
        print("Reconstruction dir: ", reconstruction_dir)
        reconstruction_score=self.clap_evaluator.audio_to_audio_similarity(source_dir,reconstruction_dir)
        print("Reconstruction score: ", reconstruction_score.item())
        
        frechet_vgg = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        frechet_vgg_score=frechet_vgg.score(source_dir,reconstruction_dir,dtype="float32")
        print("Frechet score VGGish: ", frechet_vgg_score)
        frechet_pann = FrechetAudioDistance(
            model_name="pann",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        frechet_pann_score=frechet_pann.score(source_dir,reconstruction_dir,dtype="float32")
        print("Frechet score PANN: ", frechet_pann_score)
        return reconstruction_score.item(), frechet_vgg_score, frechet_pann_score

        

    def editability_score(self,path_to_embedding,editability_prompts,method,source_dir="", n_audio_files_per_prompt=10,
                               create_audio=True,
                               random_seed=None):
        if not source_dir:
            source_dir=os.path.join(os.path.dirname(path_to_embedding),"training_audio")
        if create_audio:
            if method=="tinv":
                editability_dir=self.create_experiment_audio_tinv(path_to_embedding,editability_prompts, n_audio_files_per_prompt=n_audio_files_per_prompt,experiment_type="editability")
            elif method=="dreambooth":
                editability_dir=self.create_experiment_audio_dreambooth(path_to_embedding,editability_prompts, n_audio_files_per_prompt=n_audio_files_per_prompt,experiment_type="editability")
        else:
            editability_dir=os.path.join(os.path.dirname(path_to_embedding),"editability_audio")
        editability_scores=[]
        for subdir in os.listdir(editability_dir):
            editability_score=self.clap_evaluator.text_to_audio_similarity(os.path.join(editability_dir,subdir))
            editability_scores.append(editability_score.item())
        editability_score=np.mean(editability_scores)
        print("Editability score: ", editability_score)
        return editability_score

def evaluate_experiments(experiments_dir,
                         clap_evaluator,
                         results_csv,
                         audioldm_model_path="audioldm-m-full",
                         use_audioldm2=False,
                         reconstruction_prompts=["a recording of a {}"],
                         editability_prompts=templates.text_editability_templates,
                         n_audio_files_per_prompt=4,
                         create_audio=False,
                         random_seed=None,
                         method="tinv", # tinv or dreambooth
                         step_to_evaluate_tinv="last",
                         step_to_evaluate_dreambooth="last",
                         ):
    
    evaluator = ExperimentEvaluator(device=device, clap_evaluator=clap_evaluator,audioldm_model_path=audioldm_model_path,
                                    use_audioldm2=use_audioldm2)
    experiments_superdir=experiments_dir
    experiment_dirs=[os.path.join(experiments_superdir,dir) for dir in os.listdir(experiments_superdir) if os.path.isdir(os.path.join(experiments_superdir,dir))]
    experiment_names=[]
    reconstruction_scores=[]
    frechet_vgg_scores=[]
    frechet_pann_scores=[]
    editability_scores=[]
    for experiment_dir in experiment_dirs:
        if os.path.exists(os.path.join(experiment_dir, "reconstruction_audio")):
            shutil.rmtree(os.path.join(experiment_dir, "reconstruction_audio"))
        if os.path.exists(os.path.join(experiment_dir, "editability_audio")):
            shutil.rmtree(os.path.join(experiment_dir, "editability_audio"))
        try:
            if method=="tinv":  
            
                if step_to_evaluate_tinv=="last":
                    path_to_embedding=os.path.join(experiment_dir,"learned_embeds.bin")
                else:
                    path_to_embedding=os.path.join(experiment_dir,"learned_embeds-steps-"+str(step_to_evaluate_tinv)+".bin")
                reconstruction_score,frechet_vgg,frechet_pann = evaluator.reconstruction_score_tinv(path_to_embedding,reconstruction_prompts=reconstruction_prompts, 
                                                                                                    n_audio_files_per_prompt=n_audio_files_per_prompt,
                                                                                                    random_seed=random_seed,
                                                                                                    create_audio=create_audio)
                editability_score=evaluator.editability_score(path_to_embedding,editability_prompts=editability_prompts, 
                                                                method="tinv",
                                                                create_audio=create_audio,
                                                                n_audio_files_per_prompt=n_audio_files_per_prompt)
            elif method=="dreambooth":
                if step_to_evaluate_dreambooth=="last":
                    path_to_pipeline=os.path.join(experiment_dir,"trained_pipeline")
                else:
                    path_to_pipeline=os.path.join(experiment_dir,"pipeline_step_"+str(step_to_evaluate_dreambooth))
                reconstruction_score,frechet_vgg,frechet_pann = evaluator.reconstruction_score_dreambooth(path_to_pipeline,reconstruction_prompts=reconstruction_prompts, 
                                                                                                            n_audio_files_per_prompt=n_audio_files_per_prompt,
                                                                                                            random_seed=random_seed,
                                                                                                            create_audio=create_audio)
                editability_score=evaluator.editability_score(path_to_pipeline,editability_prompts=editability_prompts, 
                                                                method="dreambooth",
                                                                create_audio=create_audio,
                                                                n_audio_files_per_prompt=n_audio_files_per_prompt)
            experiment_name=os.path.basename(experiment_dir)
            # editability_score=0
            # scores_df=scores_df.append({"experiment_name":experiment_name,"reconstruction_score":reconstruction_score,"frechet_vgg_score":frechet_vgg,"frechet_pann_score":frechet_pann,"editability_score":0},ignore_index=True)
            experiment_names.append(experiment_name)
            reconstruction_scores.append(reconstruction_score)
            frechet_vgg_scores.append(frechet_vgg)
            frechet_pann_scores.append(frechet_pann)
            editability_scores.append(editability_score)

        except:
            print("Error in experiment: ", experiment_dir)
            continue
    scores_df=pd.DataFrame()
    scores_df["experiment_name"]=experiment_names
    scores_df["reconstruction_score"]=reconstruction_scores
    scores_df["frechet_vgg_score"]=frechet_vgg_scores
    scores_df["frechet_pann_score"]=frechet_pann_scores
    scores_df["editability_score"]=editability_scores
    scores_df.to_csv(results_csv)


if __name__ == "__main__":
    device = "cuda"
    args = parse_args()
    clap_evaluator = LAIONCLAPEvaluator(device=device,laion_clap_fusion=False, laion_clap_checkpoint=args.clap_ckpt)
    evaluate_experiments(
        args.experiment_dir,
        audioldm_model_path="audioldm2-music",
        use_audioldm2=True,
        n_audio_files_per_prompt=4,
        clap_evaluator=clap_evaluator,
        results_csv=args.results_csv,
        random_seed=42,
        step_to_evaluate_dreambooth=args.step,
        step_to_evaluate_tinv=args.step,
        method=args.method,
        create_audio=True
    )