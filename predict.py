!pip install fastapi uvicorn nest-asyncio pyngrok
!pip install numpy
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3
import joblib
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import nest_asyncio
from pyngrok import ngrok
import uvicorn

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='3OZBM72S-pFPp_u0S9Zt9H-E_0uADzRS7it17Ifo-qDM',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.direct.us-south.cloud-object-storage.appdomain.cloud')

bucket = 'indusguardai-donotdelete-pr-vrejddhl7ndkky'
object_key = 'waste_waterData.pkl'

# load data of type "application/octet-stream" into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/

streaming_body_1 = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

response = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']
model = joblib.load(io.BytesIO(response.read()))
print(model)

object_key2 = 'anomaly_waterData.pkl'
streaming_body_2 = cos_client.get_object(Bucket=bucket, Key=object_key2)['Body']
response2 = cos_client.get_object(Bucket=bucket, Key=object_key2)['Body']
model2 = joblib.load(io.BytesIO(response2.read()))
print(model2)

input_data = pd.DataFrame([{
    "pH": 0.2,
    "temperature": 99.5,
    "COD": 50,
    "BOD": 25,
    "TDS": 150,
    "lead": 0.02,
    "mercury": 0.001,
    "turbidity": 80
}])

print(model.predict(input_data))

class WaterInput(BaseModel):
    pH: float
    temperature: float
    COD: float
    BOD: float
    TDS: float
    lead: float
    mercury: float
    turbidity: float



# Allow nested event loops
nest_asyncio.apply()

# Create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"hello": "watsonx"}
ngrok.set_auth_token("2xe1X99g5eTC2JFJso3ODYln7UA_7KUJs6kuZrxi9vNukoXo8")

# Expose on public URL via ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

@app.post("/predictWaterQuality")
def predict_water_quality(data: WaterInput):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    try:
        #contamination prediction
        prediction = model.predict(input_df)[0]
        return {"contamination_prediction": int(prediction)}  # 0 or 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictAnomaly")
def predict_water_anomaly(data: WaterInput):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    try:
        # Anomaly detection
        anomaly = model2.predict(input_df)[0]
        is_anomaly = bool(anomaly == -1)  # convert to native Python bool        
        return {"is_anomaly": bool(is_anomaly)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run app
uvicorn.run(app, port=8000)
