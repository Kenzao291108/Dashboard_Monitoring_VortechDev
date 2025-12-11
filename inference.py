import json
import joblib
import csv
from collections import deque
from datetime import datetime, timedelta, timezone
import numpy as np
import paho.mqtt.client as mqtt
import ssl
import os

# ================= CONFIG =================
HIVEMQ_HOST = "73ba53f7a7ae4ddaac6a65d4deb13918.s1.eu.hivemq.cloud"
HIVEMQ_PORT = 8883
USERNAME = "admin"
PASSWORD = "Anakjb123"
USE_TLS = True

MODEL_PATH = "model_suhu.pkl"
LOG_CSV = "inference_log.csv"
TELEMETRY_TOPIC = "sensor/dht"
PRED_TOPIC = "sensor/{}/prediction"
SMOOTH_WINDOW = 5

# WIB timezone
WIB = timezone(timedelta(hours=7))
# =========================================

# Load model
model = joblib.load(MODEL_PATH)
history = {}

# ================= CSV INIT =================
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","device_id","suhu","kelembaban",
            "heat_index","pred_raw","pred_smooth","true_label"
        ])

# ================= HELPERS =================
def heat_index(T, RH):
    HI = -8.784695 + 1.61139411*T + 2.338549*RH \
         - 0.14611605*T*RH - 0.012308094*(T**2) \
         - 0.016424828*(RH**2) + 0.002211732*(T**2)*RH \
         + 0.00072546*T*(RH**2) - 0.000003582*(T**2)*(RH**2)
    return HI

def smooth(device_id, pred):
    if device_id not in history:
        history[device_id] = deque(maxlen=SMOOTH_WINDOW)
    history[device_id].append(pred)
    return max(set(history[device_id]), key=history[device_id].count)

def label_row(temp, hum):
    if temp < 28.5 and 40 <= hum <= 88.5:
        return "normal"
    if temp > 30 or (temp >= 29.5 and hum > 92.5):
        return "overheat"
    return "warning"

# ================= MQTT CALLBACK =================
def on_connect(client, userdata, flags, rc):
    print("Connected! rc=", rc)
    if rc == 0:
        print("Connection OK")
        client.subscribe(TELEMETRY_TOPIC)
    else:
        print("Connection failed")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        device_id = data.get("device_id", "esp32_default")
        T = float(data["suhu"])
        RH = float(data["kelembapan"])
        HI = heat_index(T, RH)
        true_label = label_row(T, RH)

        # prediksi model
        X = np.array([[T, RH, HI, T*RH]])
        pred_raw = model.predict(X)[0]
        pred_smooth = smooth(device_id, pred_raw)

        # publish prediksi
        pub_topic = PRED_TOPIC.format(device_id)
        client.publish(pub_topic, json.dumps({
            "prediction": str(pred_smooth),
            "timestamp": datetime.now(WIB).isoformat()
        }), qos=1)

        # simpan ke CSV
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(WIB).isoformat(),
                device_id,
                T,
                RH,
                round(HI,3),
                pred_raw,
                pred_smooth,
                true_label
            ])

        print(f"[{device_id}] T={T} RH={RH} -> pred_raw={pred_raw}, pred_smooth={pred_smooth}, true_label={true_label}")

    except Exception as e:
        print("Error processing message:", e)

# ================= MQTT CLIENT =================
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

if USE_TLS:
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

# ================= START =================
client.connect(HIVEMQ_HOST, HIVEMQ_PORT, 60)
client.loop_forever()