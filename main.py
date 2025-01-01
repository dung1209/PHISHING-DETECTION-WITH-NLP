from pydantic import BaseModel
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tensorflow.keras.models import load_model
import joblib
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles

# Khởi tạo ứng dụng FastAPI.\env\Scripts\activate
app = FastAPI()

# Gắn thư mục 'static' để phục vụ tệp tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo Jinja2Templates để render HTML
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

max_length = 128

# Tokenizer từ BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tfidf_vectorizer = joblib.load('./tfidf_vectorizer.pkl')

# Hàm encode dữ liệu nhiều dòng
def encode_texts(texts, tokenizer, max_len):
    # Đảm bảo 'texts' là một danh sách
    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return encoded['input_ids'], encoded['attention_mask']

# Xây dựng mô hình sử dụng BERT
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define a custom layer to wrap the BERT model
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model_name, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert_model_name = bert_model_name  # Store the model name instead of the model itself
        self.bert_model = TFBertModel.from_pretrained(self.bert_model_name)  # Load the model inside __init__

    def call(self, inputs):
        input_ids, attention_masks = inputs
        bert_output = self.bert_model(input_ids, attention_mask=tf.cast(attention_masks, dtype=tf.int32))
        return bert_output.last_hidden_state[:, 0, :]

    # Add get_config method to make the layer serializable
    def get_config(self):
        config = super(BertLayer, self).get_config()
        config.update({
            'bert_model_name': self.bert_model_name  # Include the model name in the config
        })
        return config
    # Add from_config method to reconstruct the layer during loading
    @classmethod
    def from_config(cls, config):
        return cls(config['bert_model_name'])

# model = tf.keras.models.load_model('./spam_classifier_model_tf.keras', custom_objects={'BertLayer': BertLayer})
model = tf.keras.models.load_model('./BERT_TFIDF.keras', custom_objects={'BertLayer': BertLayer})

# Thêm middleware CORS vào ứng dụng FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Các nguồn được phép
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa class để nhận dữ liệu tin nhắn từ người dùng
class Message(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/predict/")
async def predict_message(message: Message):
    # Mã hóa văn bản nhận từ người dùng
    test_ids, test_masks = encode_texts([message.text], tokenizer, 128)

    # Mã hóa văn bản bằng TF-IDF
    test_tfidf = tfidf_vectorizer.transform([message.text]).toarray()

    prediction = model.predict([test_ids, test_masks, test_tfidf])
    print(f"Prediction: {prediction}")

    # Xác định nhãn (Spam hoặc Ham) và độ tin cậy
    label = "Spam" if prediction[0][0] > 0.5 else "Ham"
    confidence = prediction[0][0].item() if label == "Spam" else 1 - prediction[0][0].item()

    # Trả về kết quả dự đoán
    return {
        "text": message.text,
        "prediction": label,
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
