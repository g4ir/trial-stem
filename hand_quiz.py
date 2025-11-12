import cv2
import csv
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

# --- Font tiếng Việt ---
font_path = "arial.ttf"  # Đặt file font Unicode (VD: Arial, Roboto, UVNVan.ttf) cùng thư mục
font = ImageFont.truetype(font_path, 32)

def draw_text_unicode(img, text, pos, color=(255,255,255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

class MCQ:
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
        self.userAns = None
        self.showResult = False
        self.resultTime = 0

    def draw(self, img):
        # Vẽ nền trong suốt
        overlay = img.copy()
        cv2.rectangle(overlay, (50, 50), (910, 650), (50, 50, 50), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

        img = draw_text_unicode(img, self.question, (80, 80), (255, 255, 0))
        choices = [self.choice1, self.choice2, self.choice3, self.choice4]
        bboxes = []

        for i, ch in enumerate(choices):
            x = 100 + (i % 2) * 400
            y = 250 + (i // 2) * 150
            x2, y2 = x + 300, y + 80

            # Màu mặc định
            color = (255, 255, 255)

            # Nếu đã chọn → tô đúng/sai
            if self.showResult:
                if i + 1 == self.answer:
                    color = (0, 255, 0)  # đúng → xanh
                elif i + 1 == self.userAns and self.userAns != self.answer:
                    color = (0, 0, 255)  # sai → đỏ

            cv2.rectangle(img, (x, y), (x2, y2), color, cv2.FILLED)
            img = draw_text_unicode(img, ch, (x + 20, y + 20), (0, 0, 0))
            bboxes.append((x, y, x2, y2))

        return img, bboxes

# --- Đọc dữ liệu ---
pathCSV = "Mcqs.csv"
with open(pathCSV, newline='\n', encoding='utf-8') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

mcqList = [MCQ(q) for q in dataAll]
qNo = 0
qTotal = len(dataAll)
clickCooldown = 0
score = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if clickCooldown > 0:
        clickCooldown -= 1

    if qNo < qTotal:
        mcq = mcqList[qNo]
        img, bboxes = mcq.draw(img)

        if hands and not mcq.showResult:
            lmList = hands[0]['lmList']
            cursor = lmList[8]
            length, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2])
            if length < 35 and clickCooldown == 0:
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                        mcq.userAns = i + 1
                        mcq.showResult = True
                        mcq.resultTime = time.time()
                        if mcq.userAns == mcq.answer:
                            score += 20
                        clickCooldown = 15

        # Sau 1 giây → sang câu tiếp theo
        if mcq.showResult and time.time() - mcq.resultTime > 1.2:
            qNo += 1

    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (150, 200), (800, 500), (50, 50, 50), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        img = draw_text_unicode(img, "BÀI TRẮC NGHIỆM HOÀN THÀNH", (180, 250), (255, 255, 0))
        img = draw_text_unicode(img, f"Điểm của bạn: {score}/{qTotal * 20}", (250, 350), (0, 255, 0))

    cv2.imshow("Hand Quiz - Minecraft Style", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

