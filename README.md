# Autonomous Researcher – FOSS Self-Improving

Open-source research assistant that teaches and improves itself using only free/libre libraries and compute.

## Features

- **Self-Improvement**: 
    - Runs `pylint` for code quality checks.
    - If score < 8.5, automatically formats code with [Black](https://black.readthedocs.io/).
    - No API keys required; uses only open/free tools.
- **Autonomous Research**:
    - Searches the web using DuckDuckGo and presents results with summary visualizations.
    - Caches research reports and charts for repeated queries.
- **Analytics and Monitoring**:
    - Tracks code quality, health metrics, and system resource usage.
    - Includes tools for visualizing result data and project health.

## Usage

1. **Install requirements**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the main autonomous agent**  
   ```
   python app.py
   ```
   or use Streamlit for the enhanced UI:
   ```
   streamlit run app.py
   ```

3. **Self-Improvement**  
   Run the auto-quality script:
   ```
   python agent.py
   ```

## Project Structure

- `agent.py` – Runs pylint/black and improves code quality.
- `autonomous_agent.py` – Main research agent logic.
- `charts.py` – Chart generation utilities.
- `health.py` – System resource monitoring tools.
- `quality.py` – Quality assurance (pylint/black checks).
- `requirements.txt` – Python dependencies.
- `config.py` – Configuration.
- `README.md` – This file.

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies.

## License

[MIT](LICENSE)
