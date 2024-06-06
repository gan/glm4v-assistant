# Sample GLM4V + ChatTTS AI assistant

You need a `GLM_API_KEY` to run this code. Store it in a `.env` file in the root directory of the project, or set them as environment variables.

Since glm4v can't read local images, they need to be uploaded to a server first. Here, I've configured Tencent Cloud COS.

![demo.png](demo.png)


If you are running the code on Apple Silicon, run the following command:

```
$ brew install portaudio
$ brew install miniconda

```

Create a virtual environment, update pip, and install the required packages:

```
$ git clone https://github.com/gan/glm4v-assistant.git
$ cd glm4v-assistant
$ git clone https://github.com/2noise/ChatTTS.git
```



```
$ conda create -n glm-asnt python=3.10
$ conda activate glm-asnt
```

```
$ conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing
$ pip install -U pip
$ pip install -r requirements.txt
```

Run the assistant:

```
$ python3 glm4-assistant.py
```
