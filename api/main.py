import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Global Fraud Detection Engine", version="1.0")

# Load model from the container's local directory
xgb_paysim = joblib.load('xgb_paysim_model.joblib')

class PaySimTransaction(BaseModel):
    amount: float
    balance_drop_ratio: float
    txn_velocity: int
    is_transfer_or_cashout: int
    balance_drained: int
    receiver_balance_unchanged: int

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "Fraud Detection API V1"}

@app.post("/predict/paysim")
def predict_paysim(transaction: PaySimTransaction):
    try:
        input_data = pd.DataFrame([transaction.model_dump()])
        
        expected_features = xgb_paysim.feature_names_in_
        input_data = input_data[expected_features]
        
        fraud_prob = float(xgb_paysim.predict_proba(input_data)[0][1])
        is_fraud = bool(fraud_prob >= 0.50)
        
        if fraud_prob > 0.85:
            risk_tier = "CRITICAL - FREEZE ACCOUNT"
        elif fraud_prob > 0.50:
            risk_tier = "HIGH - BLOCK TRANSACTION"
        elif fraud_prob > 0.20:
            risk_tier = "ELEVATED - REQUIRE 2FA/OTP"
        else:
            risk_tier = "LOW - APPROVE"

        return {
            "transaction_authorized": not is_fraud,
            "fraud_probability": round(fraud_prob, 4),
            "risk_tier": risk_tier,
            "model_version": "xgb_paysim_tuned_v1"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))