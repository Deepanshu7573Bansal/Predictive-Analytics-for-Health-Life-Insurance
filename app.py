import sys
import types
import numpy as np

if "sklearn.ensemble._gb_losses" not in sys.modules:
    dummy_module = types.ModuleType("sklearn.ensemble._gb_losses")

    class LeastSquaresError:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, y_true, y_pred):
            # Minimal dummy implementation for inference only.
            return ((y_true - y_pred) ** 2).mean()

    dummy_module.LeastSquaresError = LeastSquaresError
    sys.modules["sklearn.ensemble._gb_losses"] = dummy_module


import pickle
from sklearn.tree import _tree
    
def tree_from_state(*args, **kwargs):
    # Case 1: all arguments are ints – call the constructor directly.
    if all(isinstance(a, int) for a in args):
        return _tree.Tree(*args)
    
    # Case 2: assume the last argument is a state dictionary.
    state = args[-1]
    if isinstance(state, dict):
        # Retrieve or default the required parameters.
        n_features = state.get("n_features", state.get("n_features_", 0))
        n_outputs  = state.get("n_outputs", state.get("n_outputs_", 0))
        n_classes  = state.get("n_classes", 1)

        # Create the Tree instance using the extracted parameters.
        obj = _tree.Tree.__new__(_tree.Tree, n_features, n_classes, n_outputs)
        if "nodes" in state:
            nodes = state["nodes"]
            if nodes.dtype.names is not None and len(nodes.dtype.names) == 7:
                new_dtype = np.dtype({
                    'names': ['left_child', 'right_child', 'feature', 'threshold',
                              'impurity', 'n_node_samples', 'weighted_n_node_samples',
                              'missing_go_to_left'],
                    'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'],
                    'offsets': [0, 8, 16, 24, 32, 40, 48, 56],
                    'itemsize': 64
                })
                new_nodes = np.empty(nodes.shape, dtype=new_dtype)
                for name in nodes.dtype.names:
                    new_nodes[name] = nodes[name]
                new_nodes["missing_go_to_left"] = 0
                state["nodes"] = new_nodes
        obj.__setstate__(state)
        return obj
    else:
        # Fallback: if state is not a dict, assume arguments are the constructor's.
        return _tree.Tree(*args)

class CustomUnpickler(pickle.Unpickler):
    """
    A custom Unpickler that intercepts attempts to load the Tree class from
    sklearn.tree._tree, and replaces it with our tree_from_state factory.
    """
    def find_class(self, module, name):
        if module == "sklearn.tree._tree" and name == "Tree":
            return tree_from_state
        return super().find_class(module, name)

def custom_pickle_load(file_obj):
    return CustomUnpickler(file_obj).load()


import streamlit as st
import os
from tensorflow.keras.models import load_model as keras_load_model
import tensorflow as tf

def load_model(model_name):
    model_files = {
        "Linear Regression": "LinearRegression.pkl",
        "Ridge Regression": "Ridge.pkl",
        "Decision Tree Regressor": "DecisionTreeRegressor.pkl",
        "Random Forest Regressor": "RandomForestRegressor.pkl",
        "Gradient Boosting Regressor": "GradientBoostingRegressor.pkl",
        "CNN 1D": "cnn1d_model.h5",
        "FCN": "fcn_model.h5"
    }
    filename = model_files.get(model_name)
    if filename and os.path.exists(filename):
        if filename.endswith(".h5"):
            # Load a Keras model, supplying custom_objects as needed.
            custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
            return keras_load_model(filename, custom_objects=custom_objects)
        elif filename.endswith(".pkl"):
            with open(filename, "rb") as f:
                return custom_pickle_load(f)
        else:
            st.error("❌ Unsupported model file format!")
            return None
    else:
        st.error("❌ Model file not found!")
        return None

def load_scaler():
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            return pickle.load(f)
    else:
        st.error("❌ Scaler file not found!")
        return None

# Configure the Streamlit page.
st.set_page_config(page_title="💡 Health Premium Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>🩺 Health Insurance Premium Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill out the form below to estimate your insurance premium.</p>", unsafe_allow_html=True)
st.markdown("---")

# Let the user select a model.
model_name = st.selectbox("📊 Select a Machine Learning Model", [
    "Linear Regression",
    "Ridge Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "CNN 1D",
    "FCN"
])
model = load_model(model_name)
scaler = load_scaler()

# Build the input form.
with st.form("prediction_form"):
    st.subheader("📝 User Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🔢 Age", min_value=1, step=1)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        bmi = st.number_input("⚖️ BMI", min_value=0.1, format="%.2f")
        children = st.number_input("👶 Number of Children", min_value=0, step=1)
        smoker = st.selectbox("🚬 Smoker", ["Yes", "No"])
        region = st.selectbox("📜 Region", ['northeast', 'northwest', 'southeast', 'southwest'])

    with col2:
        medical_history = st.selectbox("🏥 Medical History", ["Diabetes", "Heart diseases", "High blood pressure", "No records"])
        family_medical_history = st.selectbox("👪 Family Medical History", ["Diabetes", "Heart diseases", "High blood pressure", "No records"])
        exercise_frequency = st.selectbox("🏃 Exercise Frequency", ["Frequently", "Never", "Occasionally", "Rarely"])
        occupation = st.selectbox("💼 Occupation", ["White collar", "Blue collar", "Student", "Unemployed"])
        coverage_level = st.selectbox("🛡️ Coverage Level", ["Basic", "Premium", "Standard"])

    submit_btn = st.form_submit_button("🎯 Predict Premium")

    if submit_btn and model and scaler:
        # Encode categorical inputs.
        gender_bin = 1 if gender.lower() == "male" else 0
        smoker_bin = 1 if smoker.lower() == "yes" else 0

        # Define mappings.
        region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        medical_map = {'Diabetes': 0, 'Heart diseases': 1, 'High blood pressure': 2, 'No records': 3}
        family_medical_map = {'Diabetes': 0, 'Heart diseases': 1, 'High blood pressure': 2, 'No records': 3}
        exercise_map = {'Frequently': 0, 'Never': 1, 'Occasionally': 2, 'Rarely': 3}
        occupation_map = {'Blue collar': 0, 'Student': 1, 'Unemployed': 2, 'White collar': 3}
        coverage_map = {'Basic': 0, 'Premium': 1, 'Standard': 2}

        input_data = [[
            age,
            gender_bin,
            bmi,
            children,
            smoker_bin,
            region_map[region],
            medical_map[medical_history],
            family_medical_map[family_medical_history],
            exercise_map[exercise_frequency],
            occupation_map[occupation],
            coverage_map[coverage_level]
        ]]

        # Scale the data.
        input_data_scaled = scaler.transform(input_data)

        # For deep learning models, adjust input shape.
        if model_name in ("CNN 1D", "FCN"):
            input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)

        prediction = model.predict(input_data_scaled)[0]
        prediction = float(prediction)

        st.markdown("---")
        st.success(f"💰 Estimated Health Insurance Premium: **{prediction:.2f}**")