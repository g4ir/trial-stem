import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import ImageFont, ImageDraw, Image

cv2.namedWindow("Quiz + Ve Tay", cv2.WINDOW_NORMAL)  # Cho phép resize/fullscreen
cv2.setWindowProperty("Quiz + Ve Tay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Bật full màn lúc khởi động (tùy chọn)

# --- Cấu hình webcam ---
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

# --- Font tiếng Việt ---
font_path = "arial.ttf"
font = ImageFont.truetype(font_path, 32)

# --- Dữ liệu câu hỏi ---
questions = [
    {"question": "1 + 1 = ?", "options": ["1", "2", "3", "4"], "answer": 1},
    {"question": "Thủ đô của Việt Nam là?", "options": ["Hà Nội", "Huế", "Đà Nẵng", "TP.HCM"], "answer": 0},
    {"question": "3 * 3 = ?", "options": ["6", "7", "8", "9"], "answer": 3},
]

# --- Biến điều khiển ---
qNo = 0
score = 0
selected = -1
answered = False
mode = "quiz"  # quiz / draw / end
prevX, prevY = 0, 0
drawCanvas = np.zeros((720, 960, 3), np.uint8)
lastSwitchTime = 0
drawColor = (0, 0, 255)  # màu mặc định: đỏ
hoverStartTime = [0,0,0,0]  # thời gian hover cho 4 ô câu hỏi

# --- Hàm hiển thị tiếng Việt ---
def draw_text_unicode(img, text, pos, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

# --- Hàm hiển thị câu hỏi (màu vàng đậm dần khi hover) ---
def draw_question(img, qData, selected, answered):
    global hoverStartTime
    img = draw_text_unicode(img, f"Câu {qNo+1}: {qData['question']}", (50, 50), (255, 255, 0))
    for i, opt in enumerate(qData["options"]):
        x, y = 100, 150 + i * 100
        w, h = 600, 80
        color = (255, 255, 255)  # mặc định trắng

        if answered:
            if i == qData["answer"]:
                color = (0, 255, 0)  # xanh - đúng
            elif i == selected and i != qData["answer"]:
                color = (0, 0, 255)  # đỏ - sai
        elif selected == i:
            # Hiệu ứng vàng đậm dần
            elapsed = time.time() - hoverStartTime[i]
            progress = min(1, elapsed / 2)  # 3 giây để chọn
            # từ vàng nhạt (255,255,150) → vàng đậm (0,255,255)
            r = int(255 * (1 - progress))
            g = 255
            b = int(150 * (1 - progress))
            color = (b, g, r)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        img = draw_text_unicode(img, opt, (x + 20, y + 20), (0, 0, 0))
    return img

# --- Vòng lặp chính ---
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    # --- Nút chuyển chế độ ---
    btnX, btnY, btnW, btnH = 760, 30, 170, 70
    btnColor = (70, 70, 255) if mode == "quiz" else (0, 180, 0)
    hover = False

    if hands:
        lmList = hands[0]["lmList"]
        x, y = lmList[8][0:2]
        fingers = detector.fingersUp(hands[0])
        if (btnX < x < btnX + btnW) and (btnY < y < btnY + btnH):
            hover = True
            if time.time() - lastSwitchTime > 0.5 and fingers[1] == 1:
                mode = "draw" if mode == "quiz" else "quiz"
                lastSwitchTime = time.time()

    colorHover = tuple(min(255, c + 50) for c in btnColor) if hover else btnColor
    cv2.rectangle(img, (btnX, btnY), (btnX + btnW, btnY + btnH), colorHover, -1)
    img = draw_text_unicode(img, "ĐỔI CHẾ ĐỘ", (btnX + 15, btnY + 18), (255, 255, 255))
    cv2.rectangle(img, (btnX, btnY), (btnX + btnW, btnY + btnH), (255, 255, 255), 2)

    # =========================
    # 🎯 CHẾ ĐỘ QUIZ
    # =========================
    if mode == "quiz":
        if qNo < len(questions):
            qData = questions[qNo]
            img = draw_question(img, qData, selected, answered)

            if hands:
                x, y = lmList[8][0:2]
                fingers = detector.fingersUp(hands[0])

                for i in range(4):
                    bx, by = 100, 150 + i * 100
                    bw, bh = 600, 80
                    if bx < x < bx + bw and by < y < by + bh:
                        selected = i
                        if hoverStartTime[i] == 0:
                            hoverStartTime[i] = time.time()
                        elapsed = time.time() - hoverStartTime[i]
                        if elapsed >= 2 and not answered:
                            answered = True
                            if selected == qData["answer"]:
                                score += 1
                            time.sleep(0.2)
                            qNo += 1
                            selected = -1
                            answered = False
                            hoverStartTime[i] = 0
                        break
                    else:
                        hoverStartTime[i] = 0
        else:
            mode = "end"

    # =========================
    # 🎨 CHẾ ĐỘ VẼ
    # =========================
    elif mode == "draw":
        img = draw_text_unicode(img, "CHẾ ĐỘ VẼ TAY", (30, 30), (255, 255, 0))
        cv2.rectangle(img, (20, 60), (180, 120), (50, 50, 50), -1)
        img = draw_text_unicode(img, "Xóa tất cả", (30, 75), (255, 255, 255))
        colors = {"Đỏ": (0, 0, 255), "Xanh": (0, 255, 0), "Vàng": (0, 255, 255), "Trắng": (255, 255, 255)}
        cx = 230
        for cname, cval in colors.items():
            cv2.rectangle(img, (cx, 60), (cx + 80, 120), cval, -1)
            cx += 100

        h, w, _ = img.shape
        if drawCanvas.shape[:2] != (h, w):
            drawCanvas = np.zeros((h, w, 3), np.uint8)

        if hands:
            x1, y1 = lmList[8][0:2]
            fingers = detector.fingersUp(hands[0])
            if 230 < x1 < 310 and 60 < y1 < 120 and fingers[1] == 1:
                drawColor = (0, 0, 255)
            elif 330 < x1 < 410 and 60 < y1 < 120 and fingers[1] == 1:
                drawColor = (0, 255, 0)
            elif 430 < x1 < 510 and 60 < y1 < 120 and fingers[1] == 1:
                drawColor = (0, 255, 255)
            elif 530 < x1 < 610 and 60 < y1 < 120 and fingers[1] == 1:
                drawColor = (255, 255, 255)
            if 20 < x1 < 180 and 60 < y1 < 120 and fingers[1] == 1:
                drawCanvas = np.zeros((h, w, 3), np.uint8)
            if fingers[1] == 1 and fingers[2] == 0:
                if prevX == 0 and prevY == 0:
                    prevX, prevY = x1, y1
                cv2.line(drawCanvas, (prevX, prevY), (x1, y1), drawColor, 8)
                prevX, prevY = x1, y1
            elif fingers[1] == 1 and fingers[2] == 1:
                cv2.circle(drawCanvas, (x1, y1), 40, (0, 0, 0), -1)
                prevX, prevY = 0, 0
            else:
                prevX, prevY = 0, 0

        img = cv2.addWeighted(img, 0.6, drawCanvas, 0.8, 0)

    # =========================
    # 🏁 KẾT THÚC QUIZ VỚI REPLAY
    # =========================
    elif mode == "end":
        overlay = img.copy()
        cv2.rectangle(overlay, (200, 200), (760, 450), (50, 50, 50), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        img = draw_text_unicode(img, "🎉 BÀI TRẮC NGHIỆM HOÀN THÀNH 🎉", (220, 230), (255, 255, 0))
        img = draw_text_unicode(img, f"Điểm của bạn: {score}/{len(questions)}", (320, 300), (0, 255, 0))

        # Nút Replay
        replayX, replayY, replayW, replayH = 350, 370, 250, 60
        replayColor = (100, 100, 255)
        hoverReplay = False

        if hands:
            x, y = lmList[8][0:2]
            fingers = detector.fingersUp(hands[0])
            if replayX < x < replayX + replayW and replayY < y < replayY + replayH:
                hoverReplay = True
                replayColor = (150, 150, 255)
                if fingers[1] == 1 and time.time() - lastSwitchTime > 0.5:
                    # Reset quiz
                    qNo = 0
                    score = 0
                    selected = -1
                    answered = False
                    hoverStartTime = [0,0,0,0]
                    mode = "quiz"
                    lastSwitchTime = time.time()

        cv2.rectangle(img, (replayX, replayY), (replayX + replayW, replayY + replayH), replayColor, -1)
        img = draw_text_unicode(img, "REPLAY", (replayX + 60, replayY + 15), (255, 255, 255))
        cv2.rectangle(img, (replayX, replayY), (replayX + replayW, replayY + replayH), (255, 255, 255), 2)

    cv2.imshow("Quiz + Ve Tay", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC thoát
        break
    elif key == ord('f') or key == ord('F'):  # Full screen
        cv2.setWindowProperty("Quiz + Ve Tay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('n') or key == ord('N'):  # Window bình thường
        cv2.setWindowProperty("Quiz + Ve Tay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


cap.release()
cv2.destroyAllWindows()
