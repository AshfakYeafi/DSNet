# DSNet Web Application (Brain Tumor Segmentation + Survival Prediction)

This repository contains the implementation of **DSNet**, a 3D brain tumor segmentation framework with dynamic convolutions and attention-based skip connections, along with a **Flask-based web application** for local deployment.

The web app takes **NIfTI (.nii/.nii.gz) MRI volumes** (T1, T1ce, T2, FLAIR) as input and produces segmentation masks (**WT**, **TC**, **ET**) as well as a survival prediction estimate.

⚠️ **Important Note**: The link `http://127.0.0.1:5000/` is **not a public URL**. It refers to **localhost**, meaning it only works on your own machine after running the server. To use the web application, please follow the installation and usage steps below.

---

## ✨ Features

- **Fully 3D segmentation** (UNet-like architecture with residual dynamic convolution blocks).
- **Attention-based skip connections** for improved tumor localization.
- **Survival prediction** (Random Forest + DeepSurv).
- **Web-based interface** built with Flask:
  - Upload `.nii/.nii.gz` volumes.
  - Enter patient age.
  - Generate segmentation + survival results.
- **Reproducibility**: pretrained weights, preprocessing, and scripts included.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AshfakYeafi/DSNet.git
cd DSNet
```

### 2. Create a Conda Environment (Recommended)

```bash
conda create -n dsnet python=3.9 -y
conda activate dsnet
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include Flask, PyTorch, nibabel, numpy, scikit-learn, pandas, matplotlib, shap, lifelines, etc.
See [`requirements.txt`](requirements.txt) for the full list.

### 4. Download Pretrained Weights

Place the trained DSNet weights inside the `weight/` folder:

```
weight/weight.pth
```

---

## ▶️ Running the Web Application

Start the Flask server:

```bash
flask run
```

By default, it runs on port **5000**.

* Open in your browser:
  👉 `http://127.0.0.1:5000/`

Then visit `http://<YOUR-IP>:5000/`.

---

## 🌐 Using the Web Interface

1. Upload the following MRI modalities:

   * **T1**
   * **T1ce**
   * **T2**
   * **FLAIR**
2. Enter the patient **age**.
3. Submit to generate results:

   * Segmentation masks for **WT, TC, ET** (saved in `./result/`).
   * Overlay visualizations in `static/data/`.
   * Survival prediction output (basic implementation included).

---

## 📂 Project Structure

```
├── app.py               # Flask server
├── my_model.py          # DSNet model architecture
├── requirements.txt     # Dependencies
├── weight/              # Pretrained model weights
├── static/              # Uploaded files and results
│   └── data/
├── templates/           # HTML templates for Flask
└── result/              # Segmentation outputs (.nii)
```

---

## 🛠 Troubleshooting

* **Cannot access `127.0.0.1:5000`:**

  * Ensure `python app.py` is running.
  * Open the URL in a browser on the same machine.
  * If in Docker/VM, ensure port `5000` is exposed.
* **CUDA / Memory issues:**

  * Switch to CPU by editing `device = torch.device("cpu")` in `app.py`.
  * Reduce input size if needed.
* **No outputs generated:**

  * Ensure uploaded files match the expected modality keys (`t1`, `t1ce`, `t2w`, `flair`).

---

## 🔬 Reproducibility Checklist

* [X] Training, evaluation, and inference scripts provided
* [X] Preprocessing code (center cropping to `128×128×128`)
* [X] Model weights and required file structure
* [X] SHAP scripts for survival interpretability
* [X] Web app demo with step-by-step setup

---

## 🔒 Security & Privacy

* This project uses **publicly available, de-identified BraTS datasets**.
* No personally identifiable information (PII) is included.
* For clinical deployment, additional safeguards (encryption, access control, anonymization, and compliance with GDPR/HIPAA) are required.

---

## 📖 Citation

If you use this code, please cite our paper:

```
@article{YourDSNetPaper2025,
  title   = {},
  author  = {},
  journal = {},
  year    = {}
}
```

---

## 📜 License

Specify your license here (e.g., MIT, Apache 2.0).

---

## 📬 Contact

For questions or feedback, please open a GitHub issue or contact:
📧 [yeafiashfak@gmail.com](mailto:yeafiashfak@gmail.com)
