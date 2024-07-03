# Stack Overflow Data Analysis

This project analyzes Stack Overflow data to understand trends in programming language popularity and sentiment over time utilizing pyspark.

## Setup

1. Install dependencies and binaries needed for running the project (example using Debian):
    - `sudo apt install default-jdk` (Java installation)
    - `sudo apt install python3` (Python installation)
    - `pip install -r requirements.txt` (Python dependencies, using a virtual environment in recommended)
    - Get Apache Spark from the [Official Website](https://spark.apache.org/downloads.html), extract and move Spark archive to a suitable location:
        - `tar -xvf spark-3.2.0-bin-hadoop3.2.tgz`
        - `mv spark-3.2.0-bin-hadoop3.2 /opt/spark`
    - Set Environment Variables: Edit your .bashrc or .profile file to include Spark and PySpark configurations:
        - `export SPARK_HOME=/opt/spark`
        - `export PATH=$PATH:$SPARK_HOME/bin`
        - `export PYSPARK_PYTHON=python3`

2. Place your data files (Questions.csv, Answers.csv, Tags.csv, Users.csv) in the `data/` directory. Data used was extracted from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stacksample/).

3. Make sure you have an `output/` folder created.

4. (Optional) Run the test file to check if everything is running correctly before doing the analysis on the whole dataset. This will generate a new `data-test/` folder with a sample of the original data:
    - `spark-submit test.py`
    
5. Run the main analysis:
    - `spark-submit main.py`

## Output

The analysis results and visualizations will be saved in the `output/` directory.

## Project Structure

- `src/`: Source code for data loading, analysis, and visualization
- `data/`: Input data files
- `output/`: Generated plots and analysis results
- `test.py`: Test entry point for running the analysis on a subset of the data
- `main.py`: Main entry point for running the analysis

## Dependencies

- PySpark
- Matplotlib
- WordCloud
- VADER Sentiment
- Beautiful Soup

## License

This project is licensed under the MIT License.
