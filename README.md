# Text-to-Music Personalization

#### Code for _Investigating Personalization Methods in Text to Music Generation Generation_

  

Recently, text-to-music generation models have achieved unprecedented results in synthesizing high-quality and diverse music samples from a given text prompt. Despite these advances, it remains unclear how one can generate personalized, user-specific musical concepts, manipulate them, and combine them with existing ones. For example, can one generate a rock song using their personal guitar playing style or a specific ethnic instrument? Motivated by the computer vision literature, we investigate text-to-music \textit{personalization} by exploring two established methods, namely Textual Inversion and Dreambooth. Using quantitative metrics and a user study, we evaluate their ability to reconstruct and modify new musical concepts, given only a few samples. Finally, we provide a new dataset and propose an evaluation protocol for this new task.

- [x] Release code!

- [ ] Example code for training and evaluation

- [ ] Gradio app!

- [ ] Release code for Personalized Style Transfer
  
### Install the dependencies and download AudioLDM:
  ```
pip install -r requirements.txt
git clone https://huggingface.co/cvssp/audioldm-m-full
  ```
  
  ### Training Examples
  #### DreamBooth:
  
  ```bash
export MODEL_NAME="audioldm-m-full"
export DATA_DIR="path/to/concept/audios"
export OUTPUT_DIR="path/to/output/dir"
accelerate launch dreambooth_audio.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object_class" \
  --placeholder_token="sks" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=300 \
  --learning_rate=1.0e-06 \
  --output_dir=$OUTPUT_DIR \
  --num_vectors=1 \
  --save_as_full_pipeline 
  ```

#### Textual Inversion:
```bash
export MODEL_NAME="audioldm-m-full"
export DATA_DIR="path/to/concept/audios"
export OUTPUT_DIR="path/to/output/dir"
accelerate launch --num_processes 1 --main_process_port 29603 textual_inversion_audio.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object_class" \
  --placeholder_token="<sks>" \
  --initializer="mean" \
  --initializer_token="" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir=$OUTPUT_DIR \
```


### Example Inference
```python
from pipeline.pipeline_audioldm import AudioLDMPipeline


#Textual Inversion

pipe = AudioLDMPipeline.from_pretrained("audioldm-m-full", torch_dtype=torch.float32).to("cuda")
learned_embedding = "path/to/learnedembedding"
prompt = "A recording of <sks>"
pipe.load_textual_inversion(learned_embedding)
waveform = pipe(prompt).audios

#DreamBooth
pipeline = AudioLDMPipeline.from_pretrained("path/to/dreambooth/model", torch_dtype=torch.float32).to("cuda")
prompt = "A recording of a <sks> string instrument"
waveform = pipe(prompt).audios
```
### Acknowledgments
This code is heavily  base on [AudioLDM](https://github.com/haoheliu/AudioLDM) and [Diffusers](https://github.com/huggingface/diffusers).