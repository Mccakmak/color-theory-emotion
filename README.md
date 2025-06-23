# 🎨 Emotion Assessment of YouTube Videos using Color Theory

This repository implements the research presented in:

> **Emotion Assessment of YouTube Videos using Color Theory**  
> Mert Can Cakmak, Mainuddin Shaik, Nitin Agarwal  
> *In Proceedings of the 2024 9th International Conference on Multimedia and Image Processing (ICMIP 2024)*  
> DOI: [10.1145/3665026.3665028](https://doi.org/10.1145/3665026.3665028)

---

## 🧠 Project Overview

This project investigates how dominant color patterns in YouTube videos reflect emotional undertones. Through the use of video barcodes and a custom Color-Emotion Baseline Dictionary, we decode viewer sentiment without audio or textual cues.

### Highlights
- Construction of a **Color-Emotion Baseline Dictionary** using multiple emotion wheels.
- Generation of video **color barcodes** representing dominant frame-wise colors.
- Emotion mapping using **CIEDE2000 color distance** in the **CIELAB** space.
- Emotion generalization using the **NRC Emotion Lexicon** and similarity scoring.
- Model validated on the **Trailers12k dataset** using genres and emotion alignment.

---

## 🗂 Repository Structure
📁 .idea/                         # IDE configuration files  
📁 __pycache__/                   # Python cache files  
📁 .ipynb_checkpoints/            # Jupyter autosave checkpoints  

📁 barcode_script/                # Scripts to generate video barcodes  
📁 inputs/                        # Input data (e.g., video IDs, metadata)  
📁 outputs/                       # Model outputs including emotion results  
📁 plots/                         # Visualizations like emotion distributions and barcodes  

📄 create_color_emotion_dict.py           # Builds the Color-Emotion Baseline Dictionary  
📄 emotion_clustering.py                  # Generalizes specific emotions using NRC Lexicon  
📄 find_closest_color_emotion.py          # Matches video colors to closest dictionary emotions  
📄 genre2emotion.py                       # Maps movie genres to NRC emotions  
📄 validation_genre.py                    # Validates model performance against genre expectations  
📄 mapping_words_to_emotions.py           # Maps emotion words using similarity scores  
📄 json_to_csv.py                         # Converts JSON outputs to CSV format  

📄 Emotion_Similarity_old_Glove_Bert.ipynb  # Legacy model exploration (GloVe/BERT)  
📄 NRC-Emotion-Lexicon.ipynb              # Lexicon analysis and application  

📄 color2emotion_weight_dict.pkl          # Serialized color-emotion dictionary  
📄 genre_emotion_weights.csv              # Mapping of genres to weighted emotion scores  
📄 Color_Theory paper.pdf                 # Full research paper  
📄 .gitattributes                         # Git attributes file  


## 📚 Citation

```bibtex
@inproceedings{10.1145/3665026.3665028,
  author    = {Cakmak, Mert Can and Shaik, Mainuddin and Agarwal, Nitin},
  title     = {Emotion Assessment of YouTube Videos using Color Theory},
  booktitle = {Proceedings of the 2024 9th International Conference on Multimedia and Image Processing},
  year      = {2024},
  pages     = {6--14},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3665026.3665028},
  url       = {https://doi.org/10.1145/3665026.3665028}
}
```
