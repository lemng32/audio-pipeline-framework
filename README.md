## Note
* If running for the first time, make sure you have at least 340 GB of storage available. The dataset files are 169GB and the cache for the dataset will be aproximately the same size.
* If you want to change the directory for download and caching, edit HuggingFace's environment variables as instructed [`here`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

## Setting Up
* Create HuggingFace access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens)
* Follow [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) installation instructions, make sure to accept the user conditions. No need to create another token.
* Create an access request at [`capleaf/Vivoice`](https://huggingface.co/datasets/capleaf/viVoice) to download or stream the dataset
* Modify the configuration file to include your token