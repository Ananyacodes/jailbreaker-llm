# jailbreaker-llm: Review 2 (complete project)

Setup (once):
    python3 -m venv ~/jailbreaker-venv && source ~/jailbreaker-venv/bin/activate
    pip install -r requirements.txt -r requirements-ui.txt

Run:
    python main_analysis.py          # full pipeline (original)
    streamlit run app.py             # transparent step-by-step UI
    jupyter notebook notebooks/      # 01 data exploration, 02 walkthrough
    python -m pytest tests/test_trace.py tests/test_external_data.py tests/test_scenarios.py tests/test_unsupervised.py -q

Optional real datasets (internet):
    python -m src.analysis.external_data redteam
    python -m src.analysis.external_data jbb

Slides: docs/presentations/ (11-slide short deck and 22-slide full deck)
