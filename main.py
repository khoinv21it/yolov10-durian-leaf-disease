from fastapi import FastAPI, File, UploadFile, Form, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil
import time
import uvicorn
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv
import json
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")
if not GOOGLE_AI_KEY:
    logger.error("GOOGLE_AI_KEY không được cấu hình trong .env")
    raise ValueError("GOOGLE_AI_KEY không được cấu hình trong .env")

# Mount thư mục static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Đường dẫn đến thư mục chứa mô hình
MODEL_DIR = "models"
MODELS = ['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l']

# Từ điển biện pháp phòng chống bệnh
DISEASE_RECOMMENDATIONS = {
    "Algal-Leaf-Spot": {
        "description": "Bệnh đốm tảo, do tảo Cephaleuros virescens gây ra, xuất hiện các đốm màu xanh xám hoặc cam trên lá.",
        "prevention": [
            "Loại bỏ và tiêu hủy lá bị nhiễm bệnh để giảm nguồn lây lan.",
            "Phun thuốc chứa đồng (Copper-based fungicides) như Bordeaux mixture.",
            "Tăng cường thông thoáng cho cây bằng cách tỉa cành định kỳ.",
            "Tránh tưới nước trực tiếp lên lá, đặc biệt vào buổi tối."
        ]
    },
    "Leaf-Blight": {
        "description": "Bệnh cháy lá, do nấm Rhizoctonia solani hoặc Phytophthora gây ra, khiến lá có các vết cháy nâu hoặc đen.",
        "prevention": [
            "Sử dụng thuốc trừ nấm như Mancozeb hoặc Metalaxyl theo hướng dẫn.",
            "Đảm bảo đất thoát nước tốt, tránh ngập úng.",
            "Tăng cường dinh dưỡng cho cây, đặc biệt là kali và photpho.",
            "Kiểm tra và xử lý sớm khi phát hiện dấu hiệu bệnh."
        ]
    },
    "Leaf-Spot": {
        "description": "Bệnh đốm lá, thường do nấm Cercospora hoặc Colletotrichum gây ra, tạo các đốm nhỏ màu nâu hoặc đen trên lá.",
        "prevention": [
            "Phun thuốc trừ nấm như Chlorothalonil hoặc Azoxystrobin.",
            "Loại bỏ lá rụng để giảm nguồn nấm bệnh.",
            "Trồng cây với khoảng cách hợp lý để tăng lưu thông không khí.",
            "Bón phân cân đối, tránh thừa đạm."
        ]
    },
    "No Disease": {
        "description": "Lá khỏe mạnh, không có dấu hiệu bệnh.",
        "prevention": [
            "Duy trì chăm sóc định kỳ: tưới nước, bón phân, tỉa cành.",
            "Theo dõi thường xuyên để phát hiện sớm các dấu hiệu bệnh.",
            "Sử dụng phân bón hữu cơ để tăng sức đề kháng cho cây."
        ]
    }
}

# Load mô hình YOLO
model_cache = {}
for model in MODELS:
    model_path = os.path.join(MODEL_DIR, f"{model}.pt")
    if os.path.exists(model_path):
        model_cache[model] = YOLO(model_path)
    else:
        logger.warning(f"Model file {model_path} not found!")

# Tạo thư mục uploads và results
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Cấu hình Gemini API
try:
    configure(api_key=GOOGLE_AI_KEY)
    gemini_model = GenerativeModel("gemini-1.5-flash")
except Exception as e:
    logger.error(f"Không thể cấu hình Gemini API: {str(e)}")
    raise

# Lưu lịch sử chat cho mỗi phiên
chat_histories = {}

# Prompt hệ thống để giới hạn chủ đề
SYSTEM_PROMPT = """
Bạn là một chuyên gia về bệnh lá sầu riêng và tư vấn chăm sóc sầu riêng. Chỉ trả lời các câu hỏi liên quan đến:
- Các bệnh lá sầu riêng: Algal-Leaf-Spot, Leaf-Blight, Leaf-Spot, hoặc lá khỏe mạnh (No Disease).
- Cách phòng chống, điều trị các bệnh này.
- Tư vấn chăm sóc, bón phân, tưới nước, và các biện pháp tăng sức đề kháng cho cây sầu riêng.
- Trả lời ngắn gọn đầy đủ.
Nếu câu hỏi không liên quan đến bệnh lá hoặc chăm sóc sầu riêng, hãy trả lời: "Xin lỗi, tôi chỉ hỗ trợ tư vấn về bệnh lá sầu riêng và chăm sóc sầu riêng. Vui lòng hỏi về chủ đề này!"
Sử dụng ngôn ngữ tự nhiên, thân thiện, và trả lời bằng tiếng Việt.
"""

def process_file(file_path, model_name, file_type="image"):
    if model_name not in model_cache:
        raise ValueError(f"Model {model_name} not available!")
    model = model_cache[model_name]
    
    start_time = time.time()
    if file_type == "image":
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Cannot read image file: {file_path}")
        results = model(img)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        confidences = []
        diseases = set()
        for r in results:
            boxes = r.boxes.xyxy.numpy()
            labels = r.boxes.cls.numpy()
            confs = r.boxes.conf.numpy()
            for box, label, conf in zip(boxes, labels, confs):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{model.names[int(label)]} ({conf:.2f})"
                cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                confidences.append(conf)
                diseases.add(model.names[int(label)])
        result_path = f"static/results/{model_name}_{os.path.basename(file_path)}"
        cv2.imwrite(result_path, img)
        avg_conf = np.mean(confidences) if confidences else 0
        if not diseases:
            diseases.add("No Disease")
        return result_path, avg_conf, inference_time, list(diseases)
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {file_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result_path = f"static/results/{model_name}_{os.path.basename(file_path)}"
        out = cv2.VideoWriter(result_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        confidences = []
        diseases = set()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for r in results:
                boxes = r.boxes.xyxy.numpy()
                labels = r.boxes.cls.numpy()
                confs = r.boxes.conf.numpy()
                for box, label, conf in zip(boxes, labels, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{model.names[int(label)]} ({conf:.2f})"
                    cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    confidences.append(conf)
                    diseases.add(model.names[int(label)])
            out.write(frame)
        cap.release()
        out.release()
        inference_time = (time.time() - start_time) * 1000  # ms
        avg_conf = np.mean(confidences) if confidences else 0
        if not diseases:
            diseases.add("No Disease")
        return result_path, avg_conf, inference_time, list(diseases)

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": MODELS,
        "result1": None,
        "result2": None,
        "model1": None,
        "model2": None,
        "conf1": None,
        "conf2": None,
        "time1": None,
        "time2": None,
        "diseases1": None,
        "diseases2": None,
        "recommendations": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), 
                 model1: str = Form(...), model2: str = Form(None)):
    upload_path = f"static/uploads/{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    file_type = "image" if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')) else "video"
    
    try:
        result1_path, conf1, time1, diseases1 = process_file(upload_path, model1, file_type)
        result2_path, conf2, time2, diseases2 = (None, None, None, None)
        if model2:
            result2_path, conf2, time2, diseases2 = process_file(upload_path, model2, file_type)
        
        # Gộp danh sách bệnh từ cả hai mô hình
        all_diseases = set(diseases1 or [])
        if diseases2:
            all_diseases.update(diseases2)
        recommendations = {d: DISEASE_RECOMMENDATIONS[d] for d in all_diseases}
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": MODELS,
            "error": str(e)
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": MODELS,
        "result1": f"results/{model1}_{file.filename}",
        "result2": f"results/{model2}_{file.filename}" if model2 else None,
        "model1": model1,
        "model2": model2,
        "conf1": f"{conf1:.2f}" if conf1 else None,
        "conf2": f"{conf2:.2f}" if conf2 else None,
        "time1": f"{time1:.1f}" if time1 else None,
        "time2": f"{time2:.1f}" if time2 else None,
        "diseases1": diseases1,
        "diseases2": diseases2,
        "recommendations": recommendations
    })

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(time.time())
    chat_histories[session_id] = [
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Đã hiểu! Tôi sẵn sàng tư vấn về bệnh lá sầu riêng và chăm sóc sầu riêng. Hỏi tôi bất cứ điều gì liên quan nhé!"]}
    ]
    
    try:
        while True:
            data = await websocket.receive_text()
            user_message = json.loads(data)["message"]
            
            # Thêm tin nhắn người dùng vào lịch sử
            chat_histories[session_id].append({"role": "user", "parts": [user_message]})
            
            # Gửi yêu cầu tới Gemini API
            try:
                response = await gemini_model.generate_content_async(
                    contents=chat_histories[session_id]
                )
                ai_response = response.text
            except Exception as e:
                logger.error(f"Lỗi khi gọi Gemini API: {str(e)}")
                ai_response = "Có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại!"
            
            chat_histories[session_id].append({"role": "model", "parts": [ai_response]})
            
            # Gửi phản hồi về client
            await websocket.send_text(json.dumps({"message": ai_response}))
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        await websocket.send_text(json.dumps({"message": "Kết nối bị gián đoạn. Vui lòng thử lại."}))
    finally:
        chat_histories.pop(session_id, None)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
