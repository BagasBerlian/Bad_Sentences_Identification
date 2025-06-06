# Bad Sentences Identification

This project is a web-based application designed to detect hate speech in YouTube comments. It utilizes a machine learning model to analyze comments from a given YouTube URL and identify those that contain hate speech. The application is built with Flask for the backend and uses a pre-trained SentenceTransformer model for semantic similarity comparison.

## Features

* **Hate Speech Detection**: Analyzes comments from a YouTube URL to identify potential hate speech.
* **Similarity Scoring**: Calculates a similarity score between comments and a predefined list of hate sentences.
* **Severity Level**: Classifies the detected hate speech into "Sangat Tinggi", "Tinggi", "Sedang", or "Rendah" based on the similarity score.
* **Adjustable Sensitivity**: Users can adjust the detection sensitivity (threshold) to be more or less permissive.
* **Web Interface**: Provides a user-friendly interface to input a YouTube URL and view the analysis results.
* **Sample Data Testing**: Includes a feature to test the detection model with a predefined set of sample comments.

## Technologies Used

The project is built using the following technologies and libraries:

* **Backend**: Flask
* **Machine Learning**:
    * Sentence-Transformers (`distiluse-base-multilingual-cased-v2` model)
    * Pandas
    * Numpy
    * Scikit-learn
* **Frontend**:
    * HTML/CSS/JavaScript
    * Bootstrap 5
    * Font Awesome
* **Web Scraping/API Interaction**:
    * Requests
    * BeautifulSoup4
* **Environment Variables**:
    * `python-dotenv`

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bagasberlian/bad_sentences_identification.git](https://github.com/bagasberlian/bad_sentences_identification.git)
    cd bad_sentences_identification/main
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the `main` directory and add your YouTube API key.
    ```
    YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"
    ```

## How to Run

1.  Make sure you are in the `main` directory and your virtual environment is activated.

2.  Run the Flask application:
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to `http://127.0.0.1:5000`.

## File Structure

* `app.py`: The main Flask application file containing the logic for hate speech detection and web routes.
* `kalimat_kasar.csv`: The dataset containing sentences labeled as hate speech or not.
* `requirements.txt`: A list of all the Python packages required for the project.
* `templates/index.html`: Contains the HTML template for the web interface.
* `static/`: Contains the CSS and JavaScript files for the frontend.
