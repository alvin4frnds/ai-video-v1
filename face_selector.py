import os
import random
import logging
from typing import Optional, List, Dict
from PIL import Image

class FaceSelector:
    """Utility for selecting random faces from the individual faces directory"""
    
    def __init__(self, faces_dir: str = "in/individual"):
        self.faces_dir = faces_dir
        self.selected_face = None
        self.selected_person = None
        logging.info(f"FaceSelector initialized with directory: {faces_dir}")
    
    def get_available_people(self) -> List[str]:
        """Get list of available people with face images"""
        if not os.path.exists(self.faces_dir):
            logging.warning(f"Faces directory not found: {self.faces_dir}")
            return []
        
        people = []
        for person_dir in os.listdir(self.faces_dir):
            person_path = os.path.join(self.faces_dir, person_dir)
            if os.path.isdir(person_path):
                # Check if directory has image files
                image_files = self._get_image_files(person_path)
                if image_files:
                    people.append(person_dir)
                    logging.info(f"Found {len(image_files)} images for {person_dir}")
        
        return people
    
    def _get_image_files(self, directory: str) -> List[str]:
        """Get list of image files in a directory"""
        if not os.path.exists(directory):
            return []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, file))
        
        return image_files
    
    def select_random_face(self, person: Optional[str] = None, seed: Optional[int] = None) -> Optional[str]:
        """
        Select a random face image
        
        Args:
            person: Specific person to select from, or None for random person
            seed: Random seed for reproducible selection
            
        Returns:
            Path to selected face image, or None if no faces available
        """
        if seed is not None:
            random.seed(seed)
            logging.info(f"Using seed {seed} for face selection")
        
        available_people = self.get_available_people()
        if not available_people:
            logging.error("No people with face images found")
            return None
        
        # Select person
        if person and person in available_people:
            selected_person = person
            logging.info(f"Using specified person: {person}")
        else:
            selected_person = random.choice(available_people)
            logging.info(f"Randomly selected person: {selected_person}")
        
        # Get images for selected person
        person_dir = os.path.join(self.faces_dir, selected_person)
        image_files = self._get_image_files(person_dir)
        
        if not image_files:
            logging.error(f"No images found for {selected_person}")
            return None
        
        # Select random image
        selected_image = random.choice(image_files)
        
        # Store selection for consistency
        self.selected_face = selected_image
        self.selected_person = selected_person
        
        logging.info(f"Selected face: {os.path.basename(selected_image)} from {selected_person}")
        
        # Validate image can be loaded
        try:
            with Image.open(selected_image) as img:
                logging.info(f"Face image validated: {img.size} pixels, {img.mode} mode")
        except Exception as e:
            logging.error(f"Failed to validate face image: {e}")
            return None
        
        return selected_image
    
    def get_current_selection(self) -> Dict[str, str]:
        """Get current face selection info"""
        return {
            'face_path': self.selected_face,
            'person': self.selected_person,
            'face_filename': os.path.basename(self.selected_face) if self.selected_face else None
        }
    
    def list_all_faces(self) -> Dict[str, List[str]]:
        """Get detailed list of all available faces"""
        faces_by_person = {}
        
        for person in self.get_available_people():
            person_dir = os.path.join(self.faces_dir, person)
            image_files = self._get_image_files(person_dir)
            faces_by_person[person] = [os.path.basename(f) for f in image_files]
        
        return faces_by_person