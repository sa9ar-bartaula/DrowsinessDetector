import cv2, dlib, numpy as np
from typing import Optional, Tuple

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_detector = dlib.get_frontal_face_detector()


class LipPoint:
    def extract_lip_points(self, landmarks: np.matrix, start_index: Tuple[int, int], end_index: Tuple[int, int]) -> np.ndarray:
        """Extracts the average point of a specific region of the lips."""
        
        lip_points = [landmarks[i] for i in range(start_index[0], end_index[0])] + [landmarks[i] for i in range(start_index[1], end_index[1])]
        
        return np.mean(lip_points, axis=0)

    def top_lip(self, landmarks: np.matrix) -> int:
        """Returns the average y-coordinate of the top lip."""
        try:
            top_lip_point = self.extract_lip_points(landmarks, (50, 61), (53, 64))
            return int(top_lip_point[:, 1])
        except IndexError:
            return 0 # Default value in case of an error

    def bottom_lip(self, landmarks: np.matrix) -> int:
        """Returns the average y-coordinate of the bottom lip."""
        try:
            bottom_lip_point = self.extract_lip_points(landmarks, (65, 56), (68, 59))
            return int(bottom_lip_point[:, 1])
        except IndexError:
            return 0 # Default value in case of an error



def get_landmarks(image: np.ndarray) -> Optional[np.matrix]:
    """Returns the landmarks of the first detected face in the image."""
    
    detected_faces = face_detector(image, 1)
    if len(detected_faces) != 1:  # If no face or multiple faces detected
        return None
    return np.matrix([[point.x, point.y] for point in face_predictor(image, detected_faces[0]).parts()])



def annotate_landmarks(image: np.ndarray, landmarks: np.matrix) -> np.ndarray:
    """Annotates the landmarks on the image."""
    
    annotated_image = image.copy()
    for index, point in enumerate(landmarks):
        position = (point[0, 0], point[0, 1])
        cv2.putText(annotated_image, str(index), position, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 255))
        cv2.circle(annotated_image, position, 3, color=(0, 255, 255))
    return annotated_image



def mouth_open(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """Checks if the mouth is open based on lip distance."""
    landmarks = get_landmarks(image)
    if landmarks is None:
        return image, 0
    
    annotated_image = annotate_landmarks(image, landmarks)
    lip_distance = abs(LipPoint().top_lip(landmarks) - LipPoint().bottom_lip(landmarks))
    return annotated_image, lip_distance



def update_yawn_status(lip_distance: int, prev_yawn_status: bool, yawn_count: int) -> Tuple[bool, int, str]:
    """Updates the yawn status and count."""
    
    if lip_distance > 40:
        yawn_status = True
        output_text = f" Yawn Count: {yawn_count + 1}"
    else:
        yawn_status = False
        output_text = ""

    if prev_yawn_status and not yawn_status:
        yawn_count += 1

    return yawn_status, yawn_count, output_text


def main():
    video_capture = cv2.VideoCapture(0)
    yawn_status = False
    yawn_count = 0

    while True:
        _, frame = video_capture.read()
        annotated_frame, lip_distance = mouth_open(frame)
        
        previous_yawn_status = yawn_status
        yawn_status, yawn_count, output_text = update_yawn_status(lip_distance, previous_yawn_status, yawn_count)
        
        if yawn_status:
            cv2.putText(frame, "Subject is Yawning", (50, 450), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, output_text, (50, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

        cv2.imshow('Live Landmarks', annotated_frame )
        cv2.imshow('Yawn Detection', frame )
        
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break
            
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()