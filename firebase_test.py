import firebase_admin
from firebase_admin import credentials, db

# ============================================
# FIREBASE CONFIGURATION
# ============================================

DATABASE_URL = "https://flood-watch-3862f-default-rtdb.asia-southeast1.firebasedatabase.app/"

# Load Firebase service account
cred = credentials.Certificate("serviceAccountKey.json")

# Initialize Firebase
firebase_admin.initialize_app(cred, {
    "databaseURL": DATABASE_URL
})

# ============================================
# READ CURRENT SENSOR DATA
# ============================================

ref = db.reference("/sensor_data")

data = ref.get()

print("=" * 60)
print("FIREBASE SENSOR DATA")
print("=" * 60)

if data:
    print(data)
else:
    print("No data found in /sensor_data")
