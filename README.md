# 🧠 CNN CIFAR-10 Dashboard

An interactive dashboard built with **Streamlit** and **TensorFlow** to classify images using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

---

## 🚀 Streamlit Cloud Deployment Guide (Fixing Build Errors)

If your app fails to deploy with a dependency error like:
> *Because tensorflow==2.18.0 has no wheels with a matching Python ABI tag...*

This is because Streamlit Community Cloud defaults to a Python version (e.g., Python 3.14+) that is too new for TensorFlow. You must manually select a supported Python version in the deployment settings.

### Step-by-Step Deployment:
1. Go to your **[Streamlit Community Cloud Dashboard](https://share.streamlit.io/)**.
2. If you have an existing failed deployment for this app:
   - Click the **three dots menu (...)** next to the app.
   - Select **Delete**.
3. Click **New app** (or **Deploy an app**).
4. Fill in your repository details:
   - **Repository:** `varsha-reddy18/cnn`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. **Critical Step:** Click on **"Advanced settings..."** at the bottom of the page.
6. Under the **Python version** dropdown, select **3.11** or **3.12**.
7. Click **Save** and then click **Deploy!**.

---

## 💻 Local Setup and Run

To run the application locally on your computer:

### 1. Clone the Repository
```bash
git clone https://github.com/varsha-reddy18/cnn.git
cd cnn
```

### 2. Set Up a Virtual Environment (Recommended)
Using Python 3.11 or 3.12:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
