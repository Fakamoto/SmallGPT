# 🧠 SmallGPT: A Tiny GPT Language Model Implementation 🚀

An implementation of the transformer architecture in a **Small Language Model** (SLM) 😆

SmallGPT is a compact implementation of the transformer architecture in what I call a *Small Language Model* (SLM), trained on Charles Darwin's seminal work "On the Origin of Species". 🐒

## 📈 Model Progress Showcase

### 🥴 Untrained Model Output:
```
CV.  5*etton SnN." onKpik satzas JW"am' n pVstU3rpe l.pp:n7eeS oa 8e
 KA
mFO.0f e[os,ö.t7xX.e e5 o .e.(aëp e gan.t™te0aEd;(e d rBö'ölerohr
YnsnDhö nss4ir te" 5tatmo"IlUegë ü]USZee 1 s:0te(sQfd;eseIS;ë9(hzlotco1ieH]reRea;ia)Zm6 ane Wrto --L a6 r f lnOtseriilöl"Cs"atOsae%[s7LW>e 
yNP—thee[ >lp lloEæsc
```
### 🧠 Trained Model Output:
```
importance of clearly adapted in the conditions of
life of this structure of tree procession and periods
could be
observed
by its period of food; and
the parent structure of which first and between its fact, over, argument, when it would not have perceived, yet it are bornks
```
As we can see, the trained model produces more coherent and contextually relevant text, albeit still with some imperfections. 🎉🔬

## 🏗️ Model Architecture

The SmallGPTLanguageModel model is based on the Transformer architecture. Here's a breakdown of its structure:

- 🧩 TokenEmbedding: Embeds input tokens into a continuous vector space
- 📍 PositionEmbedding: Adds positional information to the token embeddings
- 👤 Head: Implements a single attention head
- 👥 MultiHeadAttention: Combines multiple attention heads
- 🔄 FeedForward: Applies a feedforward neural network to each position
- 🧱 Block: Combines attention and feedforward layers with residual connections and layer normalization
- 🏛️ SmallGPTLanguageModel: The main model class that ties everything together

The model uses a vocabulary size of 65, an embedding dimension of 256, 6 attention heads, and 6 transformer blocks. It includes dropout for regularization and uses layer normalization. 🧮✨

## 📁 Project Structure

- 🐍 model.py: Contains the SmallGPTLanguageModel model definition
- 🏋️‍♀️ training.py: Script for training the model
- 🎨 sample.py: Script for generating text from the trained model
- 📚 input.txt: The input text file containing "On the Origin of Species"
- 💾 darwin_model.pth: The saved model weights after training

## 🚀 Getting Started

1. 📥 **Download the input text:**
   ```bash
   wget https://www.gutenberg.org/cache/epub/1228/pg1228.txt -O input.txt
   ```

2. 🏋️‍♀️ **Train the model:**
   ```bash
   python3 training.py
   ```
   Run the file as many times as you need since it will continue progress from last training. This will generate `darwin_model.pth` once training is complete.

3. 🎨 **Generate text:**
   ```bash
   python3 sample.py
   ```

## ⚠️ Limitations

Please note that this model is not fully optimized or extensively trained. It serves as a demonstration of the concepts behind transformer-based language models rather than a production-ready system. The generated text, while showing improvement over the untrained model, still contains imperfections and may not always be coherent. 🧪🔬


## 📚 Acknowledgements

- The training data for this project is sourced from the [Project Gutenberg](https://www.gutenberg.org/) website

- The neural network architecture and implementation are heavily inspired by the excellent video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy. This video provides an in-depth explanation of building a GPT model from the ground up.


## 📜 License

[MIT License](LICENSE) 🆓