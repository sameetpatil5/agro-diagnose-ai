# 🌿 AI-Powered Web Platform for Precise Plant Disease Diagnosis

![Hackathon](https://img.shields.io/badge/Hackathon-Project-blueviolet?style=for-the-badge)  
[![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)](LICENSE)  
![Tech Stack](https://img.shields.io/badge/Tech%20Stack-Python%20%7C%20Streamlit%20%7C%20Hugging%20Face%20%7C%20MongoDB%20Atlas%20%7C%20Google%20Cloud%20Storage-orange?style=for-the-badge)

---

## 📊 Problem Statement

Identifying diseases based on visual symptoms alone can be challenging, especially when symptoms are similar across multiple diseases. Traditional diagnostic methods might result in false positives or negatives, leading to improper treatment. This project aims to develop a robust machine learning-based model for disease identification using high-resolution images. The system should be trained to distinguish between different diseases and pests with high accuracy.

---

## 🌿 **Project Overview**

🌱 The agricultural sector faces challenges in diagnosing plant diseases due to overlapping symptoms and limited resources. Our **AI-Powered Web Platform** bridges this gap by leveraging cutting-edge technologies to:

- 🔍 **Identify plant diseases** from high-resolution images with remarkable accuracy.
- 💡 Provide **actionable treatment recommendations**.
- 🌍 Enable accessibility for smallholder farmers worldwide with a responsive and user-friendly design.

---

## 🚀 **Key Features**

- 📸 **Image Upload**: Upload or capture images directly via the platform.
- 🌱 **Multimodal Understanding**: Recognizes both full plant and leaf images.
- 🔍 **Categorization**: Identifies the plant type and its health status.
- ⚛️ **Intelligent Predictions**: Predicts diseases with confidence thresholds and refrains from unreliable predictions.
- 🌧️ **User Inputs**: Considers additional user-provided data (e.g., weather, watering habits).
- 📝 **Actionable Insights**: Provides treatment options if unhealthy and preventive measures if healthy.
- 🌐 **Multilingual Support**: Accessible in multiple languages.
- 🎤 **Voice Features**: Includes text-to-speech (TTS), speech-to-text (STT), and speech-to-speech (STS) for better interaction.

---

## 🔧 **Tech Stack**

| **Technology**      | **Purpose**                    |
| ------------------- | ------------------------------ |
| 🕴️ **Python**        | Core development language      |
| 🎨 **Streamlit**     | Interactive web app interface  |
| 🤗 **Hugging Face**  | AI model hosting and inference |
| ☁️ **Google Cloud**  | Image storage                  |
| 🍃 **MongoDB Atlas** | Metadata storage               |

---

## 📊 **Component Details**

- **Streamlit UI**: Provides an interactive interface for uploading images, viewing results, and interacting with the system.
- **Hugging Face Models**: Hosts models for plant/leaf classification, plant species identification, and disease detection.
- **Google Cloud**: Stores uploaded images securely.
- **MongoDB Atlas**: Stores metadata, model predictions, and user-provided information.
- **TensorFlow**: Powers the machine learning models for image classification and disease detection.

---

## 📦 **How to Run**

### Prerequisites

- Python 3.8+
- Google Cloud credentials
- MongoDB Atlas account
- Accounts for Streamlit and Hugging Face

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/plant-disease-ai.git
   cd plant-disease-ai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - `GOOGLE_CLOUD_CREDENTIALS`: Path to your Google Cloud key JSON file.
   - `MONGODB_URI`: Your MongoDB Atlas connection string.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Access the app at `http://localhost:8501`.

---

## 💪 **Contributors**

👨‍💻 **Team Name:** TECH DEVILS 👾

- **[Kartik Dani]**: [GitHub](https://github.com/Devilkd23) | [LinkedIn](https://www.linkedin.com/in/kartik-dani-06744b257)
- **[Nishant Vishwakarma]**: [GitHub]() | [LinkedIn]()
- **[Pranav Mahale]**: [GitHub]() | [LinkedIn]()
- **[Sameet Patil]**: [GitHub](https://github.com/sameetpatil5) | [LinkedIn](https://www.linkedin.com/in/sameetpatil5)
- **[Siddhant Ingole]**: [GitHub](https://github.com/siddhantingole45) | [LinkedIn](https://www.linkedin.com/in/siddhant-ingole-70b412260)
- **[Rehan Attar]**: [GitHub]() | [LinkedIn]()

---

## 📃 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🌟 **We Value Your Feedback!**

Have suggestions or ideas to improve the platform? We'd love to hear from you! Share your thoughts by reaching out to us directly or leaving a note. Together, we can revolutionize agriculture! 🌾

---

![Thank You](https://img.shields.io/badge/Thank%20You%20for%20Your%20Support-%F0%9F%8C%BF-yellow?style=for-the-badge)

---

## 🔢 **Project Progress**

### To-Do List

| Task                                | Status        | Progress |
| ----------------------------------- | ------------- | -------- |
| Set up project repository           | ✅ Completed   | 100%     |
| Design Streamlit UI                 | ⏳ In Progress | 50%      |
| Develop plant/leaf classification   | ⏳ In Progress | 40%      |
| Train plant species identification  | ❌ Not Started | 0%       |
| Implement disease detection model   | ❌ Not Started | 0%       |
| Integrate models with Streamlit UI  | ❌ Not Started | 0%       |
| Add multilingual and voice features | ❌ Not Started | 0%       |
| Test and deploy on cloud            | ❌ Not Started | 0%       |
| Collect user feedback               | ❌ Not Started | 0%       |
