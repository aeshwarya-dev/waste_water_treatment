
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

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
object_key = 'wastewater_data.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head(10)

print(df)

features = ["pH", "temperature", "COD", "BOD", "TDS", "lead", "mercury", "turbidity"]
X = df[features]
y = df["is_contaminated"]

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model to file
joblib.dump(model, "waste_waterData.pkl")
print("Model saved as waste_waterData.pkl")

from IPython.display import FileLink

FileLink('waste_waterData.pkl')

from IPython.display import HTML

#force download the pkl file
HTML('<a href="waste_waterData.pkl" download>Click here to download the model</a>')
