import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path

import streamlit as st
from joblib import load
import sklearn
@st.cache_resource
def load_trained_pipeline():
    # if the model file sits next to app.py
    model_path = Path(__file__).with_name("bigmart_best_model.joblib")
    if not model_path.exists():  # optional fallback name
        model_path = Path(__file__).with_name("bigmart_best_model.pkl")

    model = load(model_path)  # joblib.load

    # Optional: warn if scikit-learn version differs from training
    meta_path = Path(__file__).with_name("model_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        trained = meta.get("sklearn")
        if trained and trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            st.warning(
                f"Model trained with scikit-learn {trained}, "
                f"current environment has {sklearn.__version__}. "
                "Version mismatch can cause runtime errors."
            )
    return model

model = load_trained_pipeline()












# === Load Model and Version Info from Pickle ===


st.title("🛒 BigMart Sales Prediction App")
st.markdown(f"Using **scikit-learn v{sklearn.__version__}** to run predictions.")
#st.markdown(f"Using **scikit-learn v{sklearn_version}** model to predict item sales.")

# === User Inputs ===
Item_Identifier = st.text_input("Item Identifier", "FDA15")
Item_Weight = st.number_input("Item Weight", min_value=0.0, value=12.5)
Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
Item_Visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.3, step=0.01, value=0.1)
Item_Type = st.selectbox("Item Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", 
    "Health and Hygiene", "Hard Drinks", "Canned", "Breads", 
    "Starchy Foods", "Others", "Seafood"
])
Item_MRP = st.number_input("Item MRP", min_value=0.0, value=150.0)

Outlet_Identifier = st.selectbox("Outlet Identifier", [
    "OUT027", "OUT013", "OUT049", "OUT035", "OUT046", 
    "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
])
Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
Outlet_Type = st.selectbox("Outlet Type", [
    "Supermarket Type1", "Supermarket Type2", 
    "Supermarket Type3", "Grocery Store"
])
Outlet_Age = st.slider("Outlet Age (Years)", 0, 40, 15)

# === Predict Button ===
if st.button("Predict Sales"):
    input_df = pd.DataFrame([{
        "Item_Identifier": Item_Identifier,
        "Item_Weight": Item_Weight,
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": Item_Visibility,
        "Item_Type": Item_Type,
        "Item_MRP": Item_MRP,
        "Outlet_Identifier": Outlet_Identifier,
        "Outlet_Size": Outlet_Size,
        "Outlet_Location_Type": Outlet_Location_Type,
        "Outlet_Type": Outlet_Type,
        "Outlet_Age": Outlet_Age
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Item Outlet Sales: ₹{prediction:.2f}")



