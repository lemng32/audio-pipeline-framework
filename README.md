## Note
* If running for the first time, make sure you have at least 340 GB of storage available. The dataset files are 169GB and the cache for the dataset will be aproximately the same size.
* If you want to change the directory for download and caching, edit HuggingFace's environment variable as instructed [`here`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## Preparing HuggingFace Token
* Create HuggingFace access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens)
* Follow [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) installation instructions, make sure to accept the user condition. No need to create another token.
* Create an access request at [`capleaf/Vivoice`](https://huggingface.co/datasets/capleaf/viVoice) to download or stream the dataset. **It will take a day or two for the request to be accepted**.

## Setting up
1. Edit `config.json`
    - Paste your HuggingFace token into `huggingface-token` field.
    - *(Optional)* Set `out_audio_path` and `out_dataset_path` if you want custom output folders.
2. Create and activate conda environment:
    ```
    conda env create -f environment.yml
    conda activate AudioPipelineFramework
    ```
3. Run:
    ```
    python main.py
    ```
