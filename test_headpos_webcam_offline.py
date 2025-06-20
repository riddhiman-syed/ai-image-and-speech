from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

def write_to_file(pos):
    with open('assets/docs/headposition.csv','a') as f:
        now = datetime.now()
        time = now.strftime('%I:%M:%S:%p')
        date = now.strftime('%d-%B-%Y')
        f.writelines(f'\n{time}, {date}, {pos}')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def generate_frames():
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No camera")
            break
        else:
            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False

            # Get the result
            results = face_mesh.process(image)

            # To improve performance
            image.flags.writeable = True

            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    # print(y)

                    # See where the user's head tilting
                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -10:
                        text = "Looking Down"
                    else:
                        text = "Forward"

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                    cv2.line(image, p1, p2, (255, 0, 0), 2)

                    # Add the text on the image
                    cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    write_to_file(text)

            ret,buffer = cv2.imencode('.jpg',image)
            image = buffer.tobytes()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route("/")
def face_webcam():
    return render_template("face_webcam.html")

@app.route("/video")
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
