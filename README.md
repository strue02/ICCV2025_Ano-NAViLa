# Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images

Official code and resources repository for "Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images" [\[paper\]](https://arxiv.org/abs/2508.15256) (ICCV 2025) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry: (update later)

```bibtex
@inproceedings{song2025AnoNAViLa,
  title={Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images},
  author={Song, Jinsol and Wang, Jiamu and Nguyen, Anh Tien and Byeon, Keunho and Ahn, Sangjeong and Lee, Sung Hak and Kwak, Jin Tae},
  booktitle={ICCV},
  pages={000--000},
  year={2025}
}
```

## Trained Model Weights

The lymph node WSI dataset used for training cannot be publicly released at this time.  
To enable reproducibility, we provide the trained model weights as follows:

- [Original trained weights](https://drive.google.com/file/d/1ie7ch0Pcvdrf46NyPL2lyckzsuzyPoWI/view?usp=drive_link): Used to obtain the experimental results reported in the paper's tables (100 normal words and 50 abnormal words).  
- [Cleaned-word pretrained weights](https://drive.google.com/file/d/141a-w_ungtVT9e5vTszAvRonCQEGURSf/view?usp=drive_link): Re-trained using a cleaned vocabulary with duplicates removed, corresponding to the word counts reported in the paper (92 normal words and 48 abnormal words).

**Note on vocabulary sizes:**  
In the paper, we report 92 normal words and 48 abnormal words. However, the table results were generated using 100 normal words and 50 abnormal words (with duplicates). The cleaned-word model was trained later using a deduplicated vocabulary to match the reported counts.
