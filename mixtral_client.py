import requests
import json
import logging
import random
from typing import List, Dict, Optional

class MixtralClient:
    """Client for interacting with Mixtral LLM via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "mixtral"
        self.session = requests.Session()
        self.base_seed = None  # Will be set by VideoGenerator
        logging.info(f"Initialized Mixtral client with base URL: {base_url}")
    
    def check_connection(self) -> bool:
        """Check if Ollama server is running and Mixtral model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if any('mixtral' in model.lower() for model in available_models):
                    logging.info("Mixtral model found and available")
                    return True
                else:
                    logging.warning(f"Mixtral model not found. Available models: {available_models}")
                    return False
            else:
                logging.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Error checking Mixtral connection: {str(e)}")
            return False
    
    def generate_scene_prompts(self, scenes: List[str]) -> List[Dict[str, str]]:
        """Generate enhanced image prompts for each scene using Mixtral"""
        enhanced_scenes = []
        
        for i, scene in enumerate(scenes):
            logging.info(f"Generating enhanced prompt for scene {i+1}/{len(scenes)}")
            
            # Create prompt for Mixtral to enhance the scene description
            system_prompt = """You are an expert at creating detailed, visual prompts for AI image generation. 
Your task is to convert scene descriptions into highly detailed, cinematic image prompts that will create stunning visuals.

Focus on:
- Visual composition and camera angles
- Lighting and atmosphere
- Color palette and mood
- Character details and expressions
- Environmental details
- Artistic style

Keep prompts concise but descriptive (max 150 words)."""
            
            user_prompt = f"""Convert this scene into a detailed image generation prompt:

Scene: {scene}

Create a cinematic, detailed prompt that an AI image generator can use to create a compelling visual representation of this scene."""
            
            try:
                enhanced_prompt = self._call_mixtral(system_prompt, user_prompt)
                
                enhanced_scenes.append({
                    'scene_id': i + 1,
                    'original_description': scene,
                    'enhanced_prompt': enhanced_prompt,
                    'duration': 3.0,
                    'transition_type': 'fade' if i > 0 else 'none'
                })
                
                logging.info(f"Generated enhanced prompt for scene {i+1}")
                
            except Exception as e:
                logging.error(f"Failed to generate prompt for scene {i+1}: {str(e)}")
                # Fallback to basic prompt enhancement
                enhanced_scenes.append({
                    'scene_id': i + 1,
                    'original_description': scene,
                    'enhanced_prompt': self._fallback_prompt_enhancement(scene),
                    'duration': 3.0,
                    'transition_type': 'fade' if i > 0 else 'none'
                })
        
        return enhanced_scenes
    
    def _call_mixtral(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Mixtral via Ollama"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['message']['content'].strip()
        else:
            raise Exception(f"Mixtral API error: {response.status_code} - {response.text}")
    
    def _fallback_prompt_enhancement(self, scene: str) -> str:
        """Fallback prompt enhancement when Mixtral is unavailable"""
        logging.warning("Using fallback prompt enhancement")
        
        # Basic enhancement with common cinematic descriptors
        enhanced = f"{scene}, cinematic lighting, high quality, detailed, professional photography, 4k resolution, dramatic composition, vivid colors"
        
        # Add style based on scene content
        if any(word in scene.lower() for word in ['night', 'dark', 'shadow']):
            enhanced += ", moody atmosphere, dramatic shadows"
        elif any(word in scene.lower() for word in ['day', 'bright', 'sun']):
            enhanced += ", bright lighting, clear sky, vibrant colors"
        elif any(word in scene.lower() for word in ['forest', 'nature', 'tree']):
            enhanced += ", natural lighting, lush greenery, organic textures"
        elif any(word in scene.lower() for word in ['city', 'urban', 'building']):
            enhanced += ", urban environment, architectural details, street photography"
        
        return enhanced
    
    def analyze_narrative_structure(self, prompt: str) -> List[str]:
        """Use Mixtral to analyze narrative and extract optimal scene breaks"""
        
        system_prompt = """You are an expert story analyst and cinematographer. Your task is to break down narratives into optimal scenes for video generation.

Analyze the given text and identify natural scene breaks that would work well for video production. Each scene should be:
- Visually distinct and compelling with different actions, poses, or activities
- 3-5 seconds long when converted to video
- Have clear visual elements that can be captured in a still image
- Flow logically from one to the next

IMPORTANT: Generate 3-8 scenes minimum for proper video flow. For simple prompts like "woman walking in park", create multiple different actions and poses:
- Standing and looking around
- Walking along the path
- Smiling at camera
- Sitting on bench
- Posing near flowers
- Stretching or exercising
- Interacting with environment

Focus on DIFFERENT ACTIONS and POSES, not just camera angles. Each scene should show the subject doing something different.

Return ONLY a numbered list of scene descriptions, one per line. Keep each scene description concise but visually descriptive (1-2 sentences max)."""
        
        user_prompt = f"""Analyze this narrative and break it into 3-8 optimal scenes for video generation:

{prompt}

Return scenes as a simple numbered list."""
        
        try:
            if not self.check_connection():
                logging.warning("Mixtral unavailable, using fallback scene analysis")
                return self._fallback_scene_analysis(prompt)
            
            result = self._call_mixtral(system_prompt, user_prompt)
            
            # Parse the numbered list
            scenes = []
            for line in result.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    scene = line.split('.', 1)[-1].strip()
                    if scene.startswith('-'):
                        scene = scene[1:].strip()
                    if scene:
                        scenes.append(scene)
            
            logging.info(f"Mixtral identified {len(scenes)} scenes")
            
            if scenes:
                # Ensure minimum scenes for Mixtral results too
                scenes = self._ensure_minimum_scenes(scenes, prompt)
                return scenes
            else:
                return self._fallback_scene_analysis(prompt)
            
        except Exception as e:
            logging.error(f"Error in Mixtral narrative analysis: {str(e)}")
            return self._fallback_scene_analysis(prompt)
    
    def _fallback_scene_analysis(self, prompt: str) -> List[str]:
        """Fallback scene analysis when Mixtral is unavailable"""
        import re
        
        logging.info("Using fallback scene analysis")
        
        # Simple scene detection based on sentence boundaries and keywords
        sentences = re.split(r'[.!?]+', prompt.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        scenes = []
        scene_keywords = ['then', 'next', 'after', 'suddenly', 'meanwhile', 'later', 'finally']
        
        current_scene = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in scene_keywords) and current_scene:
                # Start new scene
                scenes.append(' '.join(current_scene))
                current_scene = [sentence]
            else:
                current_scene.append(sentence)
        
        if current_scene:
            scenes.append(' '.join(current_scene))
        
        # If no clear scene breaks, split into logical chunks
        if len(scenes) == 1 and len(sentences) > 3:
            mid = len(sentences) // 2
            scenes = [
                ' '.join(sentences[:mid]),
                ' '.join(sentences[mid:])
            ]
        
        # Ensure we have at least 3-8 scenes for video generation
        scenes = self._ensure_minimum_scenes(scenes, prompt)
        
        return scenes
    
    def _ensure_minimum_scenes(self, scenes: List[str], original_prompt: str) -> List[str]:
        """Ensure we have 3-8 scenes minimum for proper video generation"""
        
        if len(scenes) >= 3:
            # If we have 3 or more, cap at 8 for performance
            return scenes[:8]
        
        # If we have less than 3 scenes, expand them with different actions
        if len(scenes) == 1:
            # Single scene - create different actions/poses based on context
            base_scene = scenes[0].lower()
            expanded_scenes = self._expand_single_scene(scenes[0], base_scene)
            logging.info(f"Expanded single scene into {len(expanded_scenes)} different actions")
            return expanded_scenes
            
        elif len(scenes) == 2:
            # Two scenes - add intermediate actions
            expanded_scenes = [
                f"Starting: {scenes[0]}",
                self._create_intermediate_action(scenes[0]),
                f"Transitioning from {scenes[0]} to {scenes[1]}",
                f"Continuing: {scenes[1]}",
                self._create_final_action(scenes[1])
            ]
            logging.info("Expanded two scenes into 5 action sequences")
            return expanded_scenes
        
        return scenes
    
    def _expand_single_scene(self, original_scene: str, context: str) -> List[str]:
        """Expand a single scene into multiple actions based on context"""
        
        # Use seed for consistent randomization if available
        if self.base_seed:
            random.seed(self.base_seed)
            logging.info(f"Using seed {self.base_seed} for consistent scene expansion")
        
        # Extract key elements
        if 'walking' in context:
            if 'park' in context:
                return [
                    original_scene.replace('walking', 'standing and looking around'),
                    original_scene,  # original walking
                    original_scene.replace('walking', 'sitting on a bench'),
                    original_scene.replace('walking', 'smiling and posing near flowers'),
                    original_scene.replace('walking', 'stretching and exercising')
                ]
            elif 'street' in context or 'sidewalk' in context:
                return [
                    original_scene.replace('walking', 'standing at the corner'),
                    original_scene,  # original walking
                    original_scene.replace('walking', 'pausing to look at phone'),
                    original_scene.replace('walking', 'window shopping'),
                    original_scene.replace('walking', 'crossing the street')
                ]
            elif 'mall' in context:
                return [
                    original_scene.replace('walking', 'entering the mall'),
                    original_scene,  # original walking
                    original_scene.replace('walking', 'looking at store displays'),
                    original_scene.replace('walking', 'sitting in food court'),
                    original_scene.replace('walking', 'carrying shopping bags')
                ]
            elif 'market' in context:
                return [
                    original_scene.replace('walking', 'examining fresh produce'),
                    original_scene,  # original walking
                    original_scene.replace('walking', 'talking to vendor'),
                    original_scene.replace('walking', 'carrying market bags'),
                    original_scene.replace('walking', 'selecting items')
                ]
        
        # Generic expansion for other contexts
        subject = "person"
        if 'woman' in context:
            subject = "woman"
        elif 'man' in context:
            subject = "man"
            
        return [
            f"{subject} standing and posing",
            original_scene,  # original
            f"{subject} smiling at camera",
            f"{subject} in a different pose",
            f"{subject} in final position"
        ]
    
    def _create_intermediate_action(self, scene: str) -> str:
        """Create an intermediate action for scene transitions"""
        context = scene.lower()
        if 'walking' in context:
            return scene.replace('walking', 'pausing and looking around')
        elif 'standing' in context:
            return scene.replace('standing', 'moving slightly and adjusting pose')
        else:
            return f"Intermediate moment: {scene}"
    
    def _create_final_action(self, scene: str) -> str:
        """Create a final action for scene completion"""
        context = scene.lower()
        if 'walking' in context:
            return scene.replace('walking', 'reaching destination and turning')
        elif 'sitting' in context:
            return scene.replace('sitting', 'standing up and preparing to leave')
        else:
            return f"Final moment: {scene}"