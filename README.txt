# Medical Insurance Cost Prediction - Improved Model

Model: GradientBoostingRegressor(random_state=42)
Dataset: insurance.csv (1,338 rows)
Train/test split: 80:20, random_state=2

Measured on the held-out test split:
- Training R-squared: 0.9054 (90.54%)
- Testing R-squared: 0.8677 (86.77%)

Note: R-squared is not classification accuracy. The exact score can vary with
changes to the dataset, preprocessing, split, library versions, or model settings.
The model is an estimate, not an actual insurance quotation.

## Run the Streamlit app
1. Keep `insurance.csv`, `app.py`, and `insurance_model.pkl` in the same folder.
2. Install requirements: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Reproduce training
Open `Medical_Insurance_Cost_Prediction_Improved.ipynb` and run all cells
from top to bottom. The notebook saves `insurance_model.pkl` at the end.
