# 🧠 NeuroRex

**NeuroRex** is an intelligent head imaging diagnostic system for analyzing MRI and CT scans.  
It applies deep learning models to identify, classify, and segment neurological abnormalities such as brain tumors, hemorrhages, and strokes.

This project simulates a real-world ML system — from dataset preparation and model training, to backend APIs and potential deployment.

---

## 🚀 Current Modules

- [x] **Modality Classifier** – Detect whether input is a CT or MRI scan
- [ ] Tumor Classification (MRI)
- [ ] Tumor Segmentation (MRI)
- [ ] Hemorrhage Detection (CT)
- [ ] Stroke Segmentation (CT and FLAIR)
- [ ] CT → MRI Translation (GAN-style)

---

## 📁 Project Structure

```
NeuroRex/
├── data/                    # Local datasets (not tracked by Git)
│   └── modality_classifier/
│       ├── CT/
│       └── MRI/
├── models/                  # Model architectures and training scripts
│   └── modality_classifier/
│       └── cnn_model.py
├── notebooks/               # Prototyping and analysis
│   └── 01_modality_classifier_exploration.ipynb
├── api/                     # FastAPI backend (WIP)
├── ui/                      # Frontend (Streamlit or React)
├── mlops/                   # Experiment tracking, versioning, etc.
├── .gitignore               # Exclude data, logs, etc.
├── README.md                # Project overview (you’re here)
└── requirements.txt         # Python dependencies (coming soon)
```
---

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- FastAPI (backend, coming soon)
- Streamlit or React (frontend, WIP)
- Git + GitHub for version control
- MLflow / Docker / CI/CD (planned)

---

## 📌 Goals

- Build a realistic, multi-task ML system using real-world medical data
- Practice professional-grade project structuring, versioning, and deployment
- Demonstrate multiple CNN applications in healthcare AI
- Learn while building — not just by completing tutorials

---

## 💡 Datasets Used

- Brain Tumor Classification (MRI)
- Tumor Segmentation (BraTS)
- Hemorrhage Classification (CT)
- Stroke Segmentation (FLAIR + CT)
- Modality Classification (CT vs MRI)
- CT → MRI Translation (Paired)

All datasets are open-access and sourced from Kaggle or academic challenges.

---

## 👨‍💻 Author

Built by [Alexandru Gaitoane](https://github.com/alexandrugaitoane)  
Aspiring ML engineer focused on deep learning and medical imaging.

---

> This project is ongoing and built incrementally to simulate real-world ML development workflows.
