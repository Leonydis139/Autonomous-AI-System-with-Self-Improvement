class EnhancedAnalyticsEngine:
    def __init__(self):
        # ... (existing code) ...
        self.available_models = {
            "Random Forest": _RandomForestRegressor,
            "Linear Regression": None # Fill in if available
        }

    def predict_stock_price(self, data: pd.DataFrame, model_name="Random Forest") -> Dict:
        model_class = self.available_models.get(model_name)
        if not model_class:
            return {"error": f"Model {model_name} not available"}
        # ... (rest of your prediction code) ...
