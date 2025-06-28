from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

app = Flask(__name__)

# ============================
# Load & Preprocess Dataset
# ============================
FILENAME = 'Dataset_Model_Preprocessed.csv'  
df = pd.read_csv(FILENAME)
df = df.fillna(0)

X = df.drop(columns='spot') 
y = df['spot']

# Encoding label lokasi
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scaling RSSI value
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Train SVM Model 
# ============================
svm_model = SVC(kernel='linear', C=1, gamma=0.01, probability=True)
svm_model.fit(X_scaled, y_encoded)

# ============================
# Endpoint Prediksi Lokasi
# ============================
@app.route('/predict', methods=['POST'])
def predict_location():
    try:
        input_data = request.json
        ap_features = [input_data.get(f'ap{i+1}', -100) for i in range(X.shape[1])]
        input_df = pd.DataFrame([ap_features], columns=X.columns)

        # Scaling input sama seperti training
        input_scaled = scaler.transform(input_df)

        # Prediksi
        y_pred = svm_model.predict(input_scaled)[0]
        y_proba = svm_model.predict_proba(input_scaled)[0]

        label_pred = le.inverse_transform([y_pred])[0]
        confidence = float(np.max(y_proba))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            "estimated_location": label_pred,
            "confidence": round(confidence, 3),
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================
# Run Server
# ============================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
