# Indoor-Positioning-Model

A machine learning project to predict indoor locations (specifically for the UPATIK Building at Untirta) using preprocessed datasets and Support Vector Machine (SVM) as the core model.

---

## Run the Server

### 1. Run the Flask Server

Make sure you have installed the necessary dependencies (`flask`, `pandas`, `scikit-learn`).  
Then, run the following command in your terminal:

```bash
python server_svm.py
```
If successful, the server will run at:
```bash
http://127.0.0.1:5000/
```

### 2. Open Postman
If you don't have Postman installed, you can download it from:
`https://www.postman.com/downloads`

### 3. Test the /predict Endpoint
In Postman:
* Select POST as the request method
* and then Enter the URL:
```bash
http://127.0.0.1:5000/predict
```
* Go to the Body tab
* Choose raw and select JSON as the format
Enter input data test
```bash
{
  "ap_1": -75,
  "ap_2": -64,
  "ap_3": -22
}
```

### Expected Response Example
If the request is successful, the API will return a response like this:
```bash
{
  "confidence": 0.731,
  "estimated_location": "Lantai 1",
  "timestamp": "2025-06-28 20:32:12"
}
```
