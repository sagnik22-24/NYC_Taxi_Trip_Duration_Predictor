from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__, static_folder='templates')
app = application

@app.route('/', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            vendor_id = int(request.form.get('vendor_id')),
            pickup_datetime = request.form.get("pickup_datetime"), 
            dropoff_datetime = request.form.get("dropoff_datetime"), 
            passenger_count = int(request.form.get("passenger_count")),
            pickup_longitude = float(request.form.get("pickup_longitude")), 
            pickup_latitude = float(request.form.get("pickup_latitude")), 
            dropoff_longitude = float(request.form.get("dropoff_longitude")),
            dropoff_latitude = float(request.form.get("dropoff_latitude"))
        ) 
    new_data = data.data_transform()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("form.html", final_result = round(results/60, 2))

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug= True)

#http://127.0.0.1:5000/ in browser