from ultralytics import YOLO
import cv2
import threading
import time
import socket

from flask import Flask, jsonify, render_template_string, Response


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "plant_health_best.pt"

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.45

SERVER_PORT = 5000


# ============================================================
# LOAD AI MODEL
# ============================================================

print("==============================================")
print("        SMART PLANT MONITOR")
print("==============================================")
print()
print("Loading AI model...")

model = YOLO(MODEL_PATH)

print("AI model loaded successfully.")
print("Model classes:", model.names)
print()


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Laptop camera could not open.")
    raise SystemExit


# Optional camera resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# SHARED DATA
# ============================================================

latest_frame = None

status_lock = threading.Lock()

frame_lock = threading.Lock()


plant_status = {

    "status": "STARTING",

    "confidence": 0,

    "detections": 0,

    "last_update": "--",

    "class_name": "--"

}


# ============================================================
# GET COMPUTER IP
# ============================================================

def get_local_ip():

    try:

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        s.connect(("8.8.8.8", 80))

        ip = s.getsockname()[0]

        s.close()

        return ip

    except:

        return "YOUR-PC-IP"


# ============================================================
# AI CAMERA THREAD
# ============================================================

def camera_detection():

    global latest_frame

    print("==============================================")
    print("       AI PLANT HEALTH DETECTION")
    print("==============================================")
    print("Laptop webcam is running.")
    print("AI is monitoring the plant.")
    print()

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("Failed to read camera.")

            time.sleep(1)

            continue


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]


        # ----------------------------------------------------
        # DEFAULT VALUES
        # ----------------------------------------------------

        status = "NO LEAF DETECTED"

        confidence = 0.0

        detections_count = 0

        detected_class = ""


        # ====================================================
        # IMPORTANT
        #
        # SUPPORT BOTH:
        #
        # 1. YOLO CLASSIFICATION MODEL
        # 2. YOLO DETECTION MODEL
        # ====================================================


        # ====================================================
        # CLASSIFICATION MODEL
        # ====================================================

        if result.probs is not None:

            print("Classification model detected.")


            # -----------------------------------------------
            # TOP CLASS
            # -----------------------------------------------

            class_id = int(
                result.probs.top1
            )


            # -----------------------------------------------
            # TOP CONFIDENCE
            # -----------------------------------------------

            confidence = float(
                result.probs.top1conf
            )


            # -----------------------------------------------
            # CLASS NAME
            # -----------------------------------------------

            detected_class = model.names[class_id]


            detections_count = 1


            # -----------------------------------------------
            # CONVERT TO STATUS
            # -----------------------------------------------

            if detected_class == "Healthy_Leaf":

                status = "HEALTHY"


            elif detected_class == "Dry_Leaf":

                status = "DRY LEAF"


            elif detected_class == "Diseased_Leaf":

                status = "DISEASED"


            else:

                status = detected_class.upper()


        # ====================================================
        # OBJECT DETECTION MODEL
        # ====================================================

        elif result.boxes is not None:

            boxes = result.boxes

            detections_count = len(boxes)


            if detections_count > 0:

                confidences = (
                    boxes.conf.cpu().numpy()
                )

                class_ids = (
                    boxes.cls.cpu().numpy()
                )


                # -------------------------------------------
                # STRONGEST DETECTION
                # -------------------------------------------

                best_index = confidences.argmax()


                confidence = float(
                    confidences[best_index]
                )


                class_id = int(
                    class_ids[best_index]
                )


                detected_class = model.names[class_id]


                # -------------------------------------------
                # STATUS
                # -------------------------------------------

                if detected_class == "Healthy_Leaf":

                    status = "HEALTHY"


                elif detected_class == "Dry_Leaf":

                    status = "DRY LEAF"


                elif detected_class == "Diseased_Leaf":

                    status = "DISEASED"


                else:

                    status = detected_class.upper()


        # ====================================================
        # PRINT AI RESULT
        # ====================================================

        print(
            f"Plant: {status} | "
            f"Class: {detected_class} | "
            f"Confidence: {confidence * 100:.1f}%"
        )


        # ====================================================
        # UPDATE MOBILE DATA
        # ====================================================

        with status_lock:

            plant_status["status"] = status

            plant_status["confidence"] = round(
                confidence * 100,
                1
            )

            plant_status["detections"] = (
                detections_count
            )

            plant_status["class_name"] = (
                detected_class
                if detected_class
                else "--"
            )

            plant_status["last_update"] = (
                time.strftime("%H:%M:%S")
            )


        # ====================================================
        # DRAW YOLO RESULTS
        # ====================================================

        annotated_frame = result.plot()


        # ====================================================
        # ADD INFORMATION TO CAMERA
        # ====================================================

        cv2.rectangle(

            annotated_frame,

            (10, 10),

            (500, 125),

            (0, 0, 0),

            -1

        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        cv2.putText(

            annotated_frame,

            f"Plant: {status}",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2

        )


        # ----------------------------------------------------
        # CLASS
        # ----------------------------------------------------

        cv2.putText(

            annotated_frame,

            f"Class: {detected_class}",

            (20, 78),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        cv2.putText(

            annotated_frame,

            f"Confidence: {confidence * 100:.1f}%",

            (20, 108),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 255),

            2

        )


        # ====================================================
        # SAVE FRAME FOR MOBILE
        # ====================================================

        with frame_lock:

            latest_frame = annotated_frame.copy()


        # ====================================================
        # SHOW CAMERA ON LAPTOP
        # ====================================================

        cv2.imshow(

            "AI Plant Health - Laptop Webcam",

            annotated_frame

        )


        # ----------------------------------------------------
        # KEEP CAMERA WINDOW ALIVE
        # ----------------------------------------------------

        cv2.waitKey(1)


# ============================================================
# MOBILE LIVE VIDEO
# ============================================================

def generate_video():

    global latest_frame

    while True:

        with frame_lock:

            if latest_frame is None:

                time.sleep(0.05)

                continue

            frame = latest_frame.copy()


        # ----------------------------------------------------
        # JPEG
        # ----------------------------------------------------

        success, encoded_image = cv2.imencode(

            ".jpg",

            frame,

            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                80
            ]

        )


        if not success:

            continue


        frame_bytes = encoded_image.tobytes()


        # ----------------------------------------------------
        # SEND TO MOBILE
        # ----------------------------------------------------

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


# ============================================================
# MOBILE WEB PAGE
# ============================================================

HTML_PAGE = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Smart Plant Monitor</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    padding: 20px;

    min-height: 100vh;

    font-family: Arial, Helvetica, sans-serif;

    background:
        linear-gradient(
            135deg,
            #07140b,
            #10291a
        );

    color: white;

}


.container {

    width: 100%;

    max-width: 450px;

    margin: auto;

}


.header {

    text-align: center;

    margin-bottom: 20px;

}


.logo {

    font-size: 55px;

}


.header h1 {

    margin: 5px 0;

    font-size: 28px;

}


.header p {

    margin: 5px 0;

    color: #9eaaa2;

}


.card {

    background:
        rgba(255,255,255,0.07);

    border:
        1px solid
        rgba(255,255,255,0.1);

    border-radius: 22px;

    padding: 25px;

    box-shadow:
        0 10px 30px
        rgba(0,0,0,0.35);

}


.plant-icon {

    text-align: center;

    font-size: 65px;

}


.status {

    text-align: center;

    font-size: 31px;

    font-weight: bold;

    margin: 15px 0 25px;

}


.healthy {

    color: #42e878;

}


.dry {

    color: #ffb84d;

}


.diseased {

    color: #ff5252;

}


.waiting {

    color: #ffd43b;

}


.info {

    background:
        rgba(0,0,0,0.2);

    border-radius: 15px;

    padding: 5px 15px;

}


.row {

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 15px 0;

    border-bottom:
        1px solid
        rgba(255,255,255,0.08);

}


.row:last-child {

    border-bottom: none;

}


.label {

    color: #9da9a0;

}


.value {

    font-weight: bold;

}


.live-button {

    width: 100%;

    margin-top: 20px;

    padding: 17px;

    border: none;

    border-radius: 15px;

    background: #35c96b;

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;

}


.live-button:active {

    transform: scale(0.98);

}


#liveCamera {

    display: none;

    width: 100%;

    margin-top: 18px;

    border-radius: 15px;

    border:
        2px solid
        rgba(255,255,255,0.15);

    background: black;

}


.live-title {

    display: none;

    margin-top: 15px;

    text-align: center;

    color: #42e878;

    font-weight: bold;

}


.footer {

    text-align: center;

    margin-top: 20px;

    color: #68756d;

    font-size: 13px;

}

</style>

</head>


<body>


<div class="container">


    <!-- HEADER -->

    <div class="header">

        <div class="logo">
            🌱
        </div>

        <h1>
            Smart Plant Monitor
        </h1>

        <p>
            AI Plant Health Detection
        </p>

    </div>


    <!-- CARD -->

    <div class="card">


        <div class="plant-icon">
            🌿
        </div>


        <!-- STATUS -->

        <div
            id="status"
            class="status waiting"
        >
            STARTING
        </div>


        <!-- INFORMATION -->

        <div class="info">


            <div class="row">

                <span class="label">
                    AI Confidence
                </span>

                <span
                    class="value"
                    id="confidence"
                >
                    0%
                </span>

            </div>


            <div class="row">

                <span class="label">
                    Detected Class
                </span>

                <span
                    class="value"
                    id="className"
                >
                    --
                </span>

            </div>


            <div class="row">

                <span class="label">
                    Leaves Detected
                </span>

                <span
                    class="value"
                    id="detections"
                >
                    0
                </span>

            </div>


            <div class="row">

                <span class="label">
                    Last Update
                </span>

                <span
                    class="value"
                    id="time"
                >
                    --
                </span>

            </div>


        </div>


        <!-- ==========================================
             LIVE CAMERA BUTTON
             ========================================== -->

        <button
            id="liveButton"
            class="live-button"
            onclick="toggleCamera()"
        >

            📷 Watch Plant Live

        </button>


        <!-- LIVE TITLE -->

        <div
            id="liveTitle"
            class="live-title"
        >

            🔴 LIVE PLANT CAMERA

        </div>


        <!-- LIVE CAMERA -->

        <img
            id="liveCamera"
            alt="Live Plant Camera"
        >


    </div>


    <div class="footer">

        AI processing is running on the laptop

    </div>


</div>


<script>


// ==========================================================
// UPDATE PLANT STATUS
// ==========================================================

function updateStatus() {


    fetch("/status")

    .then(response => response.json())

    .then(data => {


        // -----------------------------------------------
        // STATUS
        // -----------------------------------------------

        const statusElement =
            document.getElementById("status");


        statusElement.innerText =
            data.status;


        // -----------------------------------------------
        // CONFIDENCE
        // -----------------------------------------------

        document.getElementById(
            "confidence"
        ).innerText =
            data.confidence + "%";


        // -----------------------------------------------
        // CLASS
        // -----------------------------------------------

        document.getElementById(
            "className"
        ).innerText =
            data.class_name;


        // -----------------------------------------------
        // DETECTIONS
        // -----------------------------------------------

        document.getElementById(
            "detections"
        ).innerText =
            data.detections;


        // -----------------------------------------------
        // TIME
        // -----------------------------------------------

        document.getElementById(
            "time"
        ).innerText =
            data.last_update;


        // -----------------------------------------------
        // REMOVE OLD COLORS
        // -----------------------------------------------

        statusElement.className =
            "status";


        // -----------------------------------------------
        // HEALTHY
        // -----------------------------------------------

        if (
            data.status === "HEALTHY"
        ) {

            statusElement.classList.add(
                "healthy"
            );

        }


        // -----------------------------------------------
        // DRY
        // -----------------------------------------------

        else if (
            data.status === "DRY LEAF"
        ) {

            statusElement.classList.add(
                "dry"
            );

        }


        // -----------------------------------------------
        // DISEASED
        // -----------------------------------------------

        else if (
            data.status === "DISEASED"
        ) {

            statusElement.classList.add(
                "diseased"
            );

        }


        // -----------------------------------------------
        // WAITING
        // -----------------------------------------------

        else {

            statusElement.classList.add(
                "waiting"
            );

        }

    })


    .catch(error => {

        document.getElementById(
            "status"
        ).innerText =
            "OFFLINE";

    });

}


// ==========================================================
// LIVE CAMERA
// ==========================================================

function toggleCamera() {


    const camera =
        document.getElementById(
            "liveCamera"
        );


    const button =
        document.getElementById(
            "liveButton"
        );


    const title =
        document.getElementById(
            "liveTitle"
        );


    // ======================================================
    // OPEN LIVE CAMERA
    // ======================================================

    if (
        camera.style.display !== "block"
    ) {


        // Important:
        // Set the image source ONLY when button is pressed.

        camera.src =
            "/video_feed?time=" +
            new Date().getTime();


        camera.style.display =
            "block";


        title.style.display =
            "block";


        button.innerText =
            "✕ Hide Live Camera";

    }


    // ======================================================
    // CLOSE LIVE CAMERA
    // ======================================================

    else {


        camera.src = "";


        camera.style.display =
            "none";


        title.style.display =
            "none";


        button.innerText =
            "📷 Watch Plant Live";

    }

}


// ==========================================================
// UPDATE EVERY SECOND
// ==========================================================

setInterval(
    updateStatus,
    1000
);


updateStatus();


</script>


</body>

</html>

"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE
    )


# ============================================================
# STATUS API
# ============================================================

@app.route("/status")
def status():

    with status_lock:

        return jsonify(
            plant_status
        )


# ============================================================
# LIVE VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_video(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # START CAMERA / AI THREAD
    # --------------------------------------------------------

    camera_thread = threading.Thread(

        target=camera_detection,

        daemon=True

    )

    camera_thread.start()


    # --------------------------------------------------------
    # GET LOCAL IP
    # --------------------------------------------------------

    local_ip = get_local_ip()


    # --------------------------------------------------------
    # SERVER INFORMATION
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("        SMART PLANT MONITOR WEB SERVER")
    print("==============================================")
    print()
    print("Open on this computer:")
    print()
    print(
        f"http://127.0.0.1:{SERVER_PORT}"
    )
    print()
    print("Open on your MOBILE:")
    print()
    print(
        f"http://{local_ip}:{SERVER_PORT}"
    )
    print()
    print("==============================================")
    print("CAMERA: Laptop webcam")
    print("AI: YOLO")
    print("LIVE CAMERA: Button controlled")
    print("CONFIDENCE: ENABLED")
    print("==============================================")
    print()


    # --------------------------------------------------------
    # FLASK
    # --------------------------------------------------------

    app.run(

        host="0.0.0.0",

        port=SERVER_PORT,

        debug=False,

        threaded=True

    )