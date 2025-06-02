
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3
from sklearn.ensemble import IsolationForest

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
object_key = 'anomaly_detection_wastewater.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_1 = pd.read_csv(body)
df_1.head(10)
features = df_1[["pH", "temperature", "COD", "BOD", "TDS", "lead", "mercury", "turbidity"]]

# 🚨 Anomaly detection using Isolation Forest
model = IsolationForest(contamination=0.2, random_state=42)
df_1["anomaly"] = model.fit_predict(features)

# 📌 -1 = anomaly, 1 = normal
print(df_1[["pH", "temperature", "anomaly"]])

# Save model to file
joblib.dump(model, "anomaly_waterData.pkl")
print("Model saved as anomaly_waterData.pkl")

from IPython.display import FileLink

FileLink('anomaly_waterData.pkl')

from IPython.display import HTML

#force download the pkl file
HTML('<a href="anomaly_waterData.pkl" download>Click here to download the model</a>')
