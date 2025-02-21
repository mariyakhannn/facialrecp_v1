import numpy as np
import cv2
import os
import pickle
import logging
import dlib
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionPipeline:
    """Pipeline for face detection, alignment, and recognition using dlib."""
    
    def __init__(
        self, 
        dataset_path: str = 'archive',
        mode: str = 'people',
        is_training: bool = True,
        image_size: Tuple[int, int] = (160, 160)
    ):
        """
        Initialize the face recognition pipeline.
        
        Args:
            dataset_path: Path to the dataset directory
            mode: Dataset mode ('people' or other)
            is_training: Whether in training mode
            image_size: Target size for face images
        """
        try:
            # Initialize dlib's face detector and facial landmarks predictor
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        except Exception as e:
            logger.error(f"Failed to initialize dlib models: {str(e)}")
            raise
        
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'lfw-deepfunneled', 'lfw-deepfunneled')
        self.mode = mode
        self.is_training = is_training
        self.image_size = image_size
        
        # Initialize face database
        self.face_database = {}
        self.load_database()

    def load_database(self):
        """Load and process the dataset based on configuration"""
        try:
            # Check if preprocessed database exists
            cache_file = f'face_database_{self.mode}_{"train" if self.is_training else "test"}.pkl'
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                logger.info("Loaded preprocessed database from cache")
                return

            # Load appropriate CSV file based on mode and training/testing
            if self.mode == 'people':
                csv_file = 'peopleDevTrain.csv' if self.is_training else 'peopleDevTest.csv'
                csv_path = os.path.join(self.dataset_path, csv_file)
                
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
                    
                people_data = pd.read_csv(csv_path)
                
                for _, row in people_data.iterrows():
                    person_name = row['name']
                    img_path = os.path.join(self.images_path, person_name, f"{person_name}_{row['images']:04d}.jpg")
                    
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Convert BGR to RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            embedding = self.get_face_embedding(img_rgb)
                            if embedding is not None:
                                if person_name not in self.face_database:
                                    self.face_database[person_name] = []
                                self.face_database[person_name].append(embedding)
                                
            # Save preprocessed database
            with open(cache_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info("Database processed and cached")
            
        except Exception as e:
            logger.error(f"Failed to load database: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """
        Detect faces in image using dlib.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of face crops and their locations
        """
        if image is None:
            raise ValueError("Image not loaded correctly")

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb_image)
            
            face_crops = []
            face_locations = []

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # Add padding and boundary checks
                x = max(0, x)
                y = max(0, y) 
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                face_img = image[y:y+h, x:x+w]
                if face_img.size > 0:  # Check if crop is valid
                    face_crops.append(face_img)
                    face_locations.append((x, y, w, h))

            return face_crops, face_locations
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return [], []

    def align_face(self, image: np.ndarray, face: dlib.rectangle) -> np.ndarray:
        """
        Align face using facial landmarks.
        
        Args:
            image: Input image
            face: Detected face rectangle
            
        Returns:
            Aligned face image
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, face)
            
            # Get coordinates of eyes
            left_eye = np.mean([(landmarks.part(36+i).x, landmarks.part(36+i).y) for i in range(6)], axis=0)
            right_eye = np.mean([(landmarks.part(42+i).x, landmarks.part(42+i).y) for i in range(6)], axis=0)
            
            # Calculate angle for alignment
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Get the center point between eyes
            eye_center = ((left_eye[0] + right_eye[0]) // 2,
                         (left_eye[1] + right_eye[1]) // 2)
            
            # Rotate the image
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            return aligned
            
        except Exception as e:
            logger.error(f"Face alignment failed: {str(e)}")
            return image

    def get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using dlib's face recognition model.
        
        Args:
            face_img: Input face image
            
        Returns:
            128-dimensional face embedding
        """
        try:
            if face_img is None or face_img.size == 0:
                return None
                
            # Convert to RGB if needed
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 4:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
            elif face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
            # Detect face and get landmarks
            faces = self.detector(face_img)
            if not faces:
                return None
                
            # Align face
            aligned_face = self.align_face(face_img, faces[0])
            
            # Get face descriptor (embedding)
            face_descriptor = self.face_encoder.compute_face_descriptor(aligned_face, self.predictor(aligned_face, faces[0]))
            return np.array(face_descriptor)
            
        except Exception as e:
            logger.error(f"Error generating face embedding: {str(e)}")
            return None

    def process_input_image(
        self, 
        image_path: str,
        verify_against: Optional[str] = None,
        threshold: float = 0.6
    ) -> Union[Dict, List[Dict]]:
        """
        Process input image for face recognition.
        
        Args:
            image_path: Path to input image
            verify_against: Optional name to verify against
            threshold: Similarity threshold
            
        Returns:
            Recognition or verification results
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
                
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Failed to load image"}
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            faces, locations = self.detect_faces(image_rgb)
            if not faces:
                return {"error": "No faces detected"}
            
            results = []
            for face_img, (x, y, w, h) in zip(faces, locations):
                # Convert face crop to RGB
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                embedding = self.get_face_embedding(face_rgb)
                if embedding is None:
                    continue
                    
                result = {
                    "location": (x, y, w, h),
                    "identity": None,
                    "confidence": 0,
                    "verified": False
                }
                
                if verify_against:
                    if verify_against in self.face_database:
                        stored_embeddings = self.face_database[verify_against]
                        similarities = [
                            np.dot(embedding, stored_emb) / (np.linalg.norm(embedding) * np.linalg.norm(stored_emb))
                            for stored_emb in stored_embeddings
                        ]
                        max_similarity = max(similarities)
                        result.update({
                            "verified": max_similarity > threshold,
                            "confidence": float(max_similarity),  # Convert to native Python float
                            "identity": verify_against if max_similarity > threshold else "Unknown"
                        })
                else:
                    # Find closest match in database
                    best_match = None
                    best_score = -1
                    
                    for person, embeddings in self.face_database.items():
                        similarities = [
                            np.dot(embedding, stored_emb) / (np.linalg.norm(embedding) * np.linalg.norm(stored_emb))
                            for stored_emb in embeddings
                        ]
                        score = max(similarities)
                        if score > best_score:
                            best_score = score
                            best_match = person
                    
                    result.update({
                        "identity": best_match if best_score > threshold else "Unknown",
                        "confidence": float(best_score)  # Convert to native Python float
                    })
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    try:
        pipeline = FaceRecognitionPipeline(
            dataset_path='archive',
            mode='people',
            is_training=True
        )
        
        while True:
            print("\nFacial Recognition System")
            print("1. Recognize faces in image")
            print("2. Verify identity")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == "1":
                try:
                    image_path = input("Enter path to image: ")
                    if not os.path.exists(image_path):
                        print(f"Error: Image file not found at {image_path}")
                        continue
                        
                    results = pipeline.process_input_image(image_path)
                    
                    if isinstance(results, dict) and "error" in results:
                        print(f"Error: {results['error']}")
                        continue
                        
                    img = cv2.imread(image_path)
                    if img is None:
                        print("Error: Failed to load image for display")
                        continue
                        
                    for result in results:
                        x, y, w, h = result["location"]
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{result['identity']} ({result['confidence']:.2f})"
                        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    cv2.imshow('Recognition Results', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                except Exception as e:
                    logger.error(f"Recognition failed: {str(e)}")
                    print(f"Recognition failed: {str(e)}")
                    
            elif choice == "2":
                try:
                    image_path = input("Enter path to image: ")
                    if not os.path.exists(image_path):
                        print(f"Error: Image file not found at {image_path}")
                        continue
                        
                    verify_name = input("Enter name to verify against: ")
                    results = pipeline.process_input_image(image_path, verify_against=verify_name)
                    
                    if isinstance(results, dict) and "error" in results:
                        print(f"Error: {results['error']}")
                        continue
                        
                    img = cv2.imread(image_path)
                    if img is None:
                        print("Error: Failed to load image for display")
                        continue
                        
                    for result in results:
                        x, y, w, h = result["location"]
                        color = (0, 255, 0) if result["verified"] else (0, 0, 255)
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        status = "Verified" if result["verified"] else "Not Verified"
                        label = f"{status} ({result['confidence']:.2f})"
                        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    cv2.imshow('Verification Results', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                except Exception as e:
                    logger.error(f"Verification failed: {str(e)}")
                    print(f"Verification failed: {str(e)}")
                    
            elif choice == "3":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        print(f"Program failed: {str(e)}")