import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request, render_template
import threading
import os

# Flask 애플리케이션 생성
app = Flask(__name__)

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose  # 포즈 감지를 위한 Mediapipe 모듈
mp_drawing = mp.solutions.drawing_utils  # 랜드마크를 그리기 위한 유틸리티

# 포즈 모델 설정 (감지 및 추적 신뢰도 설정)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 기본 카메라(카메라 인덱스 0)를 사용하여 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 상태 추적 변수
squat_count = 0  # 스쿼트 횟수를 세기 위한 변수
squat_stage = 'up'  # 현재 스쿼트 단계 추적 ("up", "down", "going_down", "going_up")

# 업로드된 비디오에서 스쿼트 횟수 세기 위한 변수
uploaded_squat_count = 0

# 각도 계산 함수
def calculate_angle(a, b, c):
    # 세 점 사이의 각도를 계산하는 함수
    # a, b, c는 세 점의 좌표를 나타냄
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점
    c = np.array(c)  # 끝 점

    # 벡터 ba와 벡터 bc 사이의 각도를 라디안으로 계산
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)  # 라디안을 도 단위로 변환

    # 각도가 [0, 180] 범위 내에 있도록 보정
    if angle > 180.0:
        angle = 360 - angle

    return angle

# 스쿼트 감지 함수
def detect_squat(cL, cR, dL, dR):
    # 양쪽 고관절과 무릎 각도를 사용하여 스쿼트를 감지하는 함수
    if cL < 130 and cR < 130 and dL < 130 and dR < 130:
        return "True"
    return "False"

def process_video():
    global squat_count, squat_stage
    while cap.isOpened():
        # 웹캠에서 프레임을 캡처
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipe에서 RGB 형식을 사용하므로 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 성능 향상을 위해 이미지 쓰기 방지

        # Mediapipe Pose를 사용하여 감지 수행
        results = pose.process(image)

        # OpenCV에서 표시하기 위해 다시 BGR로 변환
        image.flags.writeable = True  # 이미지를 다시 쓰기 가능으로 설정
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크가 감지되었을 경우 추출
        try:
            landmarks = results.pose_landmarks.landmark

            # 스쿼트 감지를 위한 좌표 추출
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # 스쿼트 감지를 위한 각도 계산
            hip_angle = calculate_angle(shoulder, hip, knee)  # 고관절 각도
            knee_angle = calculate_angle(hip, knee, ankle)  # 무릎 관절 각도
            ankle_angle = calculate_angle(knee, ankle, [ankle[0], ankle[1] + 1])  # 발목 각도 (수직 기준으로 계산)
            torso_angle = calculate_angle(hip, shoulder, [shoulder[0], shoulder[1] - 1])  # 몸통 각도 (수직 기준으로 계산)
            lumbar_pelvic_angle = calculate_angle(shoulder, hip, knee)  # 허리-골반 각도

            # 디버깅 정보 출력
            print(f"Hip Angle: {hip_angle}, Knee Angle: {knee_angle}, Ankle Angle: {ankle_angle}, Torso Angle: {torso_angle}, Lumbar-Pelvic Angle: {lumbar_pelvic_angle}, Stage: {squat_stage}")

            # 양쪽 관절에 대한 스쿼트 감지 수행
            squat_detected = detect_squat(hip_angle, hip_angle, knee_angle, knee_angle)

            # 계산된 각도를 기반으로 스쿼트 감지 로직
            if squat_stage == 'up':
                if squat_detected == "True":
                    squat_stage = 'going_down'
            elif squat_stage == 'going_down':
                if (hip_angle <= 100 or knee_angle <= 100):  # 조건 완화: 고관절 또는 무릎 각도 중 하나만 만족해도 전환
                    squat_stage = 'down'
            elif squat_stage == 'down':
                if (hip_angle >= 120 or knee_angle >= 120):  # 조건 완화: 고관절 또는 무릎 각도 중 하나만 만족해도 전환
                    squat_stage = 'going_up'
            elif squat_stage == 'going_up':
                if (hip_angle >= 160 and knee_angle >= 160 and ankle_angle >= 70 and torso_angle >= 60 and lumbar_pelvic_angle >= 130):
                    squat_count += 1
                    squat_stage = 'up'

            # 이미지에 스쿼트 횟수 표시
            cv2.putText(image, f'Squats: {squat_count}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            # 랜드마크가 감지되지 않았을 경우 오류 없이 패스
            print(f"Exception occurred: {e}")
            pass

        # 이미지에 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 프레임 표시
        cv2.imshow('Squat Counter', image)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 비디오 캡처 해제 및 모든 OpenCV 창 닫기
    cap.release()
    cv2.destroyAllWindows()

def process_uploaded_video(file_path):
    global uploaded_squat_count
    uploaded_squat_count = 0  # 초기화
    cap = cv2.VideoCapture(file_path)
    squat_stage = 'up'

    while cap.isOpened():
        # 업로드된 비디오에서 프레임을 캡처
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipe에서 RGB 형식을 사용하므로 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 성능 향상을 위해 이미지 쓰기 방지

        # Mediapipe Pose를 사용하여 감지 수행
        results = pose.process(image)

        # OpenCV에서 표시하기 위해 다시 BGR로 변환
        image.flags.writeable = True  # 이미지를 다시 쓰기 가능으로 설정
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크가 감지되었을 경우 추출
        try:
            landmarks = results.pose_landmarks.landmark

            # 스쿼트 감지를 위한 좌표 추출
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # 스쿼트 감지를 위한 각도 계산
            hip_angle = calculate_angle(shoulder, hip, knee)  # 고관절 각도
            knee_angle = calculate_angle(hip, knee, ankle)  # 무릎 관절 각도
            ankle_angle = calculate_angle(knee, ankle, [ankle[0], ankle[1] + 1])  # 발목 각도 (수직 기준으로 계산)
            torso_angle = calculate_angle(hip, shoulder, [shoulder[0], shoulder[1] - 1])  # 몸통 각도 (수직 기준으로 계산)
            lumbar_pelvic_angle = calculate_angle(shoulder, hip, knee)  # 허리-골반 각도

            # 디버깅 정보 출력
            print(f"Hip Angle: {hip_angle}, Knee Angle: {knee_angle}, Ankle Angle: {ankle_angle}, Torso Angle: {torso_angle}, Lumbar-Pelvic Angle: {lumbar_pelvic_angle}, Stage: {squat_stage}")

            # 양쪽 관절에 대한 스쿼트 감지 수행
            squat_detected = detect_squat(hip_angle, hip_angle, knee_angle, knee_angle)

            # 계산된 각도를 기반으로 스쿼트 감지 로직
            if squat_stage == 'up':
                if squat_detected == "True":
                    squat_stage = 'going_down'
            elif squat_stage == 'going_down':
                if (hip_angle <= 100 or knee_angle <= 100):  # 조건 완화: 고관절 또는 무릎 각도 중 하나만 만족해도 전환
                    squat_stage = 'down'
            elif squat_stage == 'down':
                if (hip_angle >= 120 or knee_angle >= 120):  # 조건 완화: 고관절 또는 무릎 각도 중 하나만 만족해도 전환
                    squat_stage = 'going_up'
            elif squat_stage == 'going_up':
                if (hip_angle >= 160 and knee_angle >= 160 and ankle_angle >= 70 and torso_angle >= 60 and lumbar_pelvic_angle >= 130):
                    uploaded_squat_count += 1
                    squat_stage = 'up'

            # 이미지에 스쿼트 횟수 표시
            cv2.putText(image, f'Squats: {uploaded_squat_count}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            # 랜드마크가 감지되지 않았을 경우 오류 없이 패스
            print(f"Exception occurred: {e}")
            pass

        # 이미지에 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 프레임 표시
        cv2.imshow('Uploaded Squat Counter', image)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 비디오 캡처 해제 및 모든 OpenCV 창 닫기
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['GET'])
def start_camera():
    threading.Thread(target=process_video).start()
    return jsonify({'message': 'Camera processing started'}), 200

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        threading.Thread(target=process_uploaded_video, args=(file_path,)).start()
        return jsonify({'message': 'Video processing started'}), 200

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)