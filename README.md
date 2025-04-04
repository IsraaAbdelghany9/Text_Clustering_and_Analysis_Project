
# Text Clustering and Analysis Project

This project is a comprehensive text clustering and analysis pipeline that processes text data, applies clustering algorithms, evaluates the results, and visualizes the clusters. It also provides an API for predicting the cluster of new text data using FastAPI.

---

## Features

- **Text Preprocessing**: Tokenization, stopword removal, lemmatization, and cleaning of text data.
- **Feature Extraction**: Converts text data into TF-IDF features for clustering.
- **Clustering**: Implements K-Means and hierarchical clustering algorithms.
- **Evaluation**: Computes Silhouette Score and Purity Score to evaluate clustering performance.
- **Visualization**: Provides visualizations such as the Elbow Method, Silhouette Analysis, t-SNE plots, and dendrograms.
- **API Deployment**: Exposes clustering functionality via a FastAPI RESTful API.
- **Dockerized Deployment**: Easily deployable using Docker.

---

## Project Structure

```
Text_Clustering_and_Analysis_Project
├── Full_project_in1Notebook
│   ├── Full_Project_Notebook.ipynb                   # Full project workflow
│   └── utils.py                                      # Functions implementation
├── Full_Structured_project
│   ├── Dockerfile                                    # Docker configuration
│   ├── notebooks
│   │   └── EDA.ipynb                                 # Exploratory Data Analysis
│   ├── requirements.txt                              # Python dependencies
│   ├── results                                       # Directory for saving models and results
│   │   ├── kmeans_model.joblib
│   │   ├── labels.joblib
│   │   ├── linkage_matrix.joblib
│   │   ├── purity_score.joblib
│   │   ├── silhouette.joblib
│   │   └── vectorizer.joblib
│   └── src
│       ├── clustering.py                            # Clustering algorithms
│       ├── evaluation.py                            # Evaluation metrics
│       ├── feature_extraction.py                    # TF-IDF vectorization
│       ├── main.py                                  # Main script for clustering and evaluation
│       ├── preprocessing.py                         # Text preprocessing functions
│       ├── server.py                                # FastAPI server for API
│       ├── test.py
│       └── visualization.py                         # Visualization functions
├── LICENSE                                          # Project License
└── README.md                                        # Project documentation
```
---

## Requirements

- Python 3.10 or higher
- Libraries:
  - [`numpy`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fevaluation.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A7%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FMl2-Project%2FReading_data.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A29%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2FFull_Project_Notebook.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A24%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fnotebooks%2FEDA.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A29%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fclustering.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A9%2C%22character%22%3A11%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fvisualization.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A14%2C%22character%22%3A11%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")
  - [`matplotlib`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fvisualization.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A7%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A1%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2FFull_Project_Notebook.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A37%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fnotebooks%2FEDA.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A462%2C%22character%22%3A12%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")
  - [`pandas`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FMl2-Project%2FReading_data.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A28%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2FFull_Project_Notebook.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fnotebooks%2FEDA.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A28%2C%22character%22%3A12%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")
  - `scikit-learn`
  - [`nltk`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fpreprocessing.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FMl2-Project%2FReading_data.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A16%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A4%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2FFull_Project_Notebook.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A33%2C%22character%22%3A12%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fnotebooks%2FEDA.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A16%2C%22character%22%3A12%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")
  - [`scipy`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fclustering.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A1%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fvisualization.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A5%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2FFull_Project_Notebook.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A59%2C%22character%22%3A10%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fnotebooks%2FEDA.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A40%2C%22character%22%3A10%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Ffeature_extraction.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A8%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")
  - [`uvicorn`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A6%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2FDockerfile%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A22%2C%22character%22%3A6%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")
  - [`fastapi`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fserver.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A7%2C%22character%22%3A0%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition")

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## Datasets

1. **20 Newsgroups Dataset**:
   - A dataset of newsgroup posts categorized into topics such as `talk.religion.misc`, `comp.graphics`, and `sci.space`.
   - Loaded using [`sklearn.datasets.fetch_20newsgroups`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fevaluation.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A1%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Ffeature_extraction.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A5%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fclustering.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fvisualization.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A1%2C%22character%22%3A5%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition").

---

## Running the Project

### 1. Run the Main Script
The [`main.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "/home/israa/Desktop/NLP_projects/ML_2_proj/src/main.py") script performs the following:
- Preprocesses the text data.
- Extracts TF-IDF features.
- Applies K-Means and hierarchical clustering.
- Evaluates clustering performance.
- Saves the models and results to the `results/` directory.

Run the script:
```bash
python src/main.py
```

### 2. Run the FastAPI Server
The [`server.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fserver.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "/home/israa/Desktop/NLP_projects/ML_2_proj/src/server.py") script provides an API for predicting the cluster of new text data and fetching clustering results.

Start the server:
```bash
uvicorn src.server:app --reload
```

Access the API at `http://127.0.0.1:8000`.

### 3. Run with Docker
Build and run the Docker container:
```bash
docker build -t text-clustering-app .
docker run -p 8000:8000 text-clustering-app
```

---

## API Endpoints

### `/`
- **Method**: GET
- **Description**: Returns a welcome message.

### `/results`
- **Method**: GET
- **Description**: Fetches stored clustering results (Silhouette Score and Purity Score).

### `/predict`
- **Method**: POST
- **Description**: Predicts the cluster of new text data.
- **Request Body**:
  ```json
  {
    "text": "Your text here"
  }
  ```
- **Response**:
  ```json
  {
    "text": "Your text here",
    "predicted_cluster": 2
  }
  ```

---

## Visualizations

The project provides the following visualizations:
1. **Elbow Method**: Helps determine the optimal number of clusters.
2. **Silhouette Analysis**: Evaluates clustering quality.
3. **t-SNE Visualization**: Visualizes clusters in 2D space.
4. **Dendrogram**: Visualizes hierarchical clustering.

---

## Evaluation Metrics

1. **Silhouette Score**:
   - Measures how similar an object is to its own cluster compared to other clusters.
   - Higher scores indicate better-defined clusters.

2. **Purity Score**:
   - Measures the extent to which clusters contain a single class.

---

## Development

### Adding New Features
1. Add your logic to the appropriate module in the [`src/`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fserver.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A4%2C%22character%22%3A5%7D%7D%5D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "Go to definition") directory.
2. Expose the functionality via an API endpoint in [`server.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fisraa%2FDesktop%2FNLP_projects%2FML_2_proj%2Fsrc%2Fserver.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22b1fdfebb-820a-41cf-a2cf-12ff5fbabc1f%22%5D "/home/israa/Desktop/NLP_projects/ML_2_proj/src/server.py").

### Running Tests
To add and run tests, use `pytest`:
```bash
pip install pytest
pytest
```

---

## License

This project is licensed under the  Apache2.0 License. See the `LICENSE` file for details.

---

## Acknowledgments

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)

---

Let me know if you need further modifications!






