#Sentiment-Classfication-and-Analysis


## Description

The multi-task transformer model leverages a pre-trained transformer from Hugging Face to perform two NLP tasks simultaneously:

1. **Sentence Classification:** Classifies sentences into predefined categories (e.g., Positive, Negative, Neutral).
2. **Sentiment Analysis:** Analyzes the sentiment of sentences, identifying them as either Positive or Negative.

## Setup and Installation

### Prerequisites

- Python 3.8
- Docker (for containerized setup)
- Git

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NLP_Project.git
   cd NLP_Project
   
2. **Create a virtual environment and activate it:**
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

3. **Install the dependencies**
pip install -r requirements.txt

**Docker Setup**
1.**Build the Docker image:**
docker build -t nlp_project .

2. **Run the Docker container:**
   docker run --rm nlp_project

**Train the model**
python src/train.py

**Test the model**
python src/test.py

(optional)
**Running the training code inside the Docker Container**
docker run --rm nlp_project



**Continuous Integration with GitHub Actions**
This repository includes a GitHub Actions workflow for building and pushing the Docker image to GitHub Container Registry (ghcr.io) on every push to the repository. The workflow file is located at .github/workflows/deploy-image.yml.

GitHub Actions Workflow
The workflow:

1. Checks out the repository.
2. Logs in to the GitHub Container Registry.
3. Extracts metadata for Docker.
4. Builds and pushes the Docker image.
