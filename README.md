# 🏛️ AIML Capstone: Preserving Heritage & Enhancing Tourism with AI

This capstone project is divided into two major components, each focused on applying AI/ML to real-world tourism and heritage conservation challenges:

🔹 **Part 1 — Image Classification using Deep Learning (TensorFlow)**  
Classify historical structures based on images to assist government agencies in automated heritage monitoring.

🔹 **Part 2 — Tourism Data Analysis & Recommendation System**  
Analyze user behaviour and build a collaborative filtering model to recommend places to tourists.

---

## 📁 Project Structure

AIML-Capstone-Heritage-Tourism/
│
├── data/ # Small datasets included in repo
│ ├── user.csv
│ ├── tourism_with_id.csv
│ └── tourism_rating.csv
│
├── docs/ # Problem statement or notes
│ └── problem_statement.pdf
│
├── notebooks/ # Jupyter / Colab notebooks
│ ├── Capstone_Part1-Image_Classification.ipynb
│ └── Capstone_Part2-Tourism_Recommendation_System.ipynb
│
├── README.md # Project overview & instructions
└── requirements.txt # Python dependencies


**Note:** The large Part 1 dataset (~132 MB) is not included in the repo due to GitHub size limits. 
Download from Google Drive: [Download Part 1 Dataset](https://drive.google.com/uc?export=download&id=1S9SMIGewX0mk_-2Umanc41nNsZ5Koutt)  
Extract the zip into `data/` before running Part 1 notebook.

---

## 🧠 Part 1 — Deep Learning: Heritage Structures Image Classification

**🎯 Objective**  
Build a TensorFlow-based image classification model using transfer learning to categorize historical structures into predefined classes.

**🔍 Steps Performed**  

1️⃣ **Dataset Preparation**  
- `structures_dataset.zip` → training images  
- `dataset_test/` → testing images  
- Checked class distribution and image counts  

2️⃣ **Visualization**  
- Plotted 8–10 sample images per class using OpenCV  

3️⃣ **Transfer Learning Setup**  
- Pre-trained CNN backbone: VGG16  
- Loaded ImageNet weights  
- Froze convolutional layers  
- Added Dense layers + ReLU, Dropout, Softmax output  

4️⃣ **Model Compilation**  
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metric: Accuracy  

5️⃣ **Early Stopping**  
- Callback to stop training when validation accuracy plateaued  

6️⃣ **Model Training**  
- Without Augmentation → baseline performance  
- With Augmentation → rotation, zoom, shear, flips, width/height shifts  

7️⃣ **Evaluation**  
- Plotted training vs validation accuracy & loss  
- Checked underfitting, overfitting, convergence  

**📌 Outcome (Part 1)**  
- Trained CNN model achieved strong validation accuracy  
- Demonstrated practical feasibility for heritage structure classification  

---

## 🗺️ Part 2 — Tourism Recommendation System (Collaborative Filtering)

**🎯 Objective**  
Analyze tourism data and build an Item-Based Collaborative Filtering model to recommend tourist spots similar to a selected location.

**📚 Datasets Used**  
- `user.csv` → Tourist demographics  
- `tourism_with_id.csv` → Details of 437 tourist attractions  
- `tourism_rating.csv` → Ratings by 300 users  

**🔍 Data Analysis Performed**  
- User analysis: age distribution, cities visited, rating patterns  
- Location & category analysis: popular tourist spot categories, best cities for different types of tourists  
- Combined dataset insights: most loved cities, highest-rated attractions, preferred categories  

**🤖 Collaborative Filtering Recommendation Model**  
- Created 300×437 user-item rating matrix  
- Calculated item-based cosine similarity  

**Recommendation function:**
```python
def recommend_places(place_name, similarity_df, n=5):
    if place_name not in similarity_df.columns:
        return f"No data available for {place_name}"
    
    sim_scores = similarity_df[place_name].drop(place_name)
    top_recommendations = sim_scores.sort_values(ascending=False).head(n)

    recs_df = pd.DataFrame({
        'Recommended Place': top_recommendations.index,
        'Similarity Score': top_recommendations.values
    }).reset_index(drop=True)

    return recs_df
Visualization function:

python
Copy code
def plot_recommendations(place_name, similarity_df, n=5):
    recs = recommend_places(place_name, similarity_df, n)
    print(f"Top {n} recommendations for '{place_name}':\n")
    print(recs.to_string(index=False))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.barh(recs['Recommended Place'], recs['Similarity Score'], color='skyblue')
    plt.xlabel("Similarity Score")
    plt.title(f"Top {n} Similar Places to {place_name}")
    plt.gca().invert_yaxis()
    plt.show()
📊 Example Output
Recommendations for "Monumen Nasional":

Recommended Place	Similarity Score
Wisata Mangrove Tapak	0.2688
Danau Rawa Pening	0.2627
Museum Sonobudoyo Unit I	0.2601
Dunia Fantasi	0.2524
Situ Patenggang	0.2411

🧠 Insights & Learnings
Tourists prefer Amusement Parks and Nature Attractions

Similar tourist spots can be predicted using rating behavior

Collaborative Filtering effectively recommends destinations

🚀 Future Enhancements
Hybrid recommendation model (content + CF)

Incorporate user context (age, city, budget)

Neural Collaborative Filtering (NCF)

Interactive dashboard

⚙️ Requirements
Install dependencies:

bash
Copy code
pip install -r requirements.txt
requirements.txt contents:

ini
Copy code
tensorflow==2.19.0
keras==3.10.0
numpy==2.0.2
pandas==2.2.2
matplotlib==3.10.0
opencv-python==4.12.0
Pillow==11.3.0
scikit-learn==1.6.1
🙌 Author
Priyanka Pabla
AIML Postgraduate Program – Capstone Project

🎉 End of Project