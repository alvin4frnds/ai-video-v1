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
        """Generate enhanced image prompts for each scene using Mixtral with consistent character"""
        enhanced_scenes = []
        
        # First, generate a detailed character description for consistency
        character_profile = self._generate_character_profile(scenes)
        logging.info(f"Generated character profile for consistency across {len(scenes)} scenes")
        logging.info(f"Character: {character_profile[:100]}...")
        
        for i, scene in enumerate(scenes):
            logging.info(f"Generating enhanced prompt for scene {i+1}/{len(scenes)}")
            
            # Create prompt for Mixtral to enhance the scene description
            system_prompt = f"""You are an expert at creating detailed, visual prompts for AI image generation. 
Your task is to convert scene descriptions into highly detailed, cinematic image prompts featuring a CONSISTENT CHARACTER.

CHARACTER PROFILE (use this EXACT character in ALL scenes):
{character_profile}

IMPORTANT CONSTRAINTS:
- ALWAYS use the EXACT character described above
- Maintain complete visual consistency (same person, same clothing, same appearance)
- NEVER change the character's appearance, clothing, or features
- Add negative prompts to prevent multiple people or character variations

Focus on:
- Visual composition and camera angles for the consistent character
- Lighting and atmosphere
- Color palette and mood
- Character expressions and poses (keeping same appearance)
- Environmental details
- Artistic style

Respond in this JSON format:
{{
  "positive_prompt": "detailed positive prompt featuring the exact character (max 150 words)",
  "negative_prompt": "multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person, different clothing, different hair, different appearance"
}}

Keep prompts concise but descriptive."""
            
            user_prompt = f"""Convert this scene into a detailed image generation prompt featuring the CONSISTENT CHARACTER:

CHARACTER TO USE: {character_profile}

Scene: {scene}

Create a cinematic, detailed prompt featuring the EXACT character described above in this scene. The character's appearance, clothing, and features must remain identical to the profile. Include comprehensive negative prompts to prevent multiple people or character variations.

Respond with JSON containing both positive_prompt and negative_prompt fields."""
            
            try:
                response = self._call_mixtral(system_prompt, user_prompt)
                
                # Try to parse JSON response
                try:
                    prompt_data = json.loads(response)
                    positive_prompt = prompt_data.get('positive_prompt', response)
                    negative_prompt = prompt_data.get('negative_prompt', 'multiple people, crowd, group, two people, three people, many people')
                except json.JSONDecodeError:
                    # Fallback if not JSON
                    positive_prompt = response
                    negative_prompt = 'multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person'
                
                enhanced_scenes.append({
                    'scene_id': i + 1,
                    'original_description': scene,
                    'enhanced_prompt': positive_prompt,
                    'negative_prompt': negative_prompt,
                    'duration': 3.0,
                    'transition_type': 'fade' if i > 0 else 'none'
                })
                
                logging.info(f"Generated enhanced prompt for scene {i+1}")
                logging.info(f"  Positive: {positive_prompt[:80]}...")
                logging.info(f"  Negative: {negative_prompt[:80]}...")
                
            except Exception as e:
                logging.error(f"Failed to generate prompt for scene {i+1}: {str(e)}")
                # Fallback to basic prompt enhancement with character profile
                fallback_data = self._fallback_prompt_enhancement(scene, character_profile)
                enhanced_scenes.append({
                    'scene_id': i + 1,
                    'original_description': scene,
                    'enhanced_prompt': fallback_data['positive_prompt'],
                    'negative_prompt': fallback_data['negative_prompt'],
                    'duration': 3.0,
                    'transition_type': 'fade' if i > 0 else 'none'
                })
        
        return enhanced_scenes
    
    def _generate_character_profile(self, scenes: List[str]) -> str:
        """Generate a detailed character description for consistency across all scenes"""
        
        # Analyze scenes to determine character type
        all_scenes_text = " ".join(scenes)
        
        system_prompt = """You are an expert character designer for AI image generation. 
Create a highly detailed, consistent character description that will be used across multiple scenes in a video.

Your task is to describe ONE person in great detail including:
- Physical appearance (age, ethnicity, height, build)
- Facial features (eyes, hair, skin tone, distinctive features)
- Clothing and style (outfit, colors, accessories, footwear)
- Overall aesthetic and personality impression

Be specific enough that an AI can generate the same person consistently across different scenes.
Keep the description concise but comprehensive (max 100 words).

Respond with just the character description, no additional text."""

        user_prompt = f"""Based on these scene descriptions, create a detailed character profile for the main person who will appear throughout:

Scenes: {all_scenes_text}

Create a comprehensive character description focusing on visual consistency for AI image generation."""
        
        try:
            character_description = self._call_mixtral(system_prompt, user_prompt)
            logging.info("Successfully generated character profile with Mixtral")
            return character_description.strip()
        except Exception as e:
            logging.warning(f"Failed to generate character profile with Mixtral: {str(e)}")
            # Fallback character profile
            return self._fallback_character_profile(all_scenes_text)
    
    def _fallback_character_profile(self, scenes_text: str) -> str:
        """Generate fallback character profile when Mixtral is unavailable"""
        logging.warning("Using fallback character profile generation")
        
        # Basic character based on scene content
        if any(word in scenes_text.lower() for word in ['woman', 'she', 'her', 'girl', 'lady']):
            character = "Young woman in her mid-20s, shoulder-length brown hair, warm brown eyes, fair complexion, wearing a casual red dress with white sneakers, friendly and approachable demeanor, 5'6\" height, slim build"
        elif any(word in scenes_text.lower() for word in ['man', 'he', 'his', 'guy', 'male']):
            character = "Young man in his late 20s, short dark hair, blue eyes, medium complexion, wearing casual jeans and navy blue t-shirt with white sneakers, confident posture, 5'10\" height, athletic build"
        else:
            # Default to woman if unclear
            character = "Young woman in her mid-20s, shoulder-length brown hair, warm brown eyes, fair complexion, wearing a casual red dress with white sneakers, friendly and approachable demeanor, 5'6\" height, slim build"
        
        return character
    
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
    
    def _fallback_prompt_enhancement(self, scene: str, character_profile: str = "") -> Dict[str, str]:
        """Fallback prompt enhancement when Mixtral is unavailable"""
        logging.warning("Using fallback prompt enhancement")
        
        # Ensure single person language
        scene_single = scene.replace(" people ", " person ").replace(" they ", " she ").replace(" them ", " her ")
        
        # Build prompt with character profile if available
        if character_profile:
            positive_prompt = f"{character_profile}, {scene_single}, consistent character, same appearance, cinematic lighting, high quality, detailed, professional photography, 4k resolution, dramatic composition, vivid colors"
        else:
            positive_prompt = f"single person, {scene_single}, one person only, cinematic lighting, high quality, detailed, professional photography, 4k resolution, dramatic composition, vivid colors"
        
        # Add style based on scene content
        if any(word in scene.lower() for word in ['night', 'dark', 'shadow']):
            positive_prompt += ", moody atmosphere, dramatic shadows"
        elif any(word in scene.lower() for word in ['day', 'bright', 'sun']):
            positive_prompt += ", bright lighting, clear sky, vibrant colors"
        elif any(word in scene.lower() for word in ['forest', 'nature', 'tree']):
            positive_prompt += ", natural lighting, lush greenery, organic textures"
        elif any(word in scene.lower() for word in ['city', 'urban', 'building']):
            positive_prompt += ", urban environment, architectural details, street photography"
        
        # Comprehensive negative prompt for single person constraint and character consistency
        negative_prompt = "multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person, several people, couple, pair, twins, family, team, friends, strangers, bystanders, different clothing, different hair, different appearance, changed outfit, different person"
        
        return {
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt
        }
    
    def analyze_narrative_structure(self, prompt: str) -> List[str]:
        """Use Mixtral to analyze narrative and extract optimal scene breaks"""
        
        system_prompt = """You are an expert story analyst and cinematographer. Your task is to break down narratives into optimal scenes for video generation.

Analyze the given text and identify natural scene breaks that would work well for video production. Each scene should be:
- Visually distinct and compelling with different actions, poses, or activities
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

CRITICAL: You must respond with ONLY valid JSON in this exact format:
{
  "scenes": [
    {
      "id": 1,
      "description": "Scene description here",
      "duration": 3.0,
      "transition": "fade"
    },
    {
      "id": 2,
      "description": "Next scene description",
      "duration": 2.5,
      "transition": "fade"
    }
  ]
}

Duration should be 2.0-5.0 seconds based on scene complexity. Use "fade", "cut", or "dissolve" for transitions.
Return ONLY the JSON, no other text."""
        
        user_prompt = f"""Analyze this narrative and break it into 3-8 optimal scenes for video generation:

{prompt}

Return ONLY valid JSON with the exact format specified above."""
        
        try:
            if not self.check_connection():
                logging.warning("Mixtral unavailable, using fallback scene analysis")
                return self._fallback_scene_analysis(prompt)
            
            result = self._call_mixtral(system_prompt, user_prompt)
            
            # Parse JSON response
            try:
                # Clean up the response - sometimes Mixtral adds extra text
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    data = json.loads(json_str)
                    
                    if 'scenes' in data and len(data['scenes']) > 0:
                        scenes = []
                        for scene_data in data['scenes']:
                            scene_info = {
                                'description': scene_data.get('description', ''),
                                'duration': float(scene_data.get('duration', 3.0)),
                                'transition': scene_data.get('transition', 'fade')
                            }
                            scenes.append(scene_info)
                        
                        logging.info(f"Mixtral provided {len(scenes)} scenes with timing data")
                        for i, scene in enumerate(scenes):
                            logging.info(f"Scene {i+1}: {scene['description'][:80]}... (duration: {scene['duration']}s, transition: {scene['transition']})")
                        
                        return scenes
                    else:
                        logging.warning("Mixtral JSON missing scenes array")
                        return self._fallback_scene_analysis_with_timing(prompt)
                else:
                    logging.warning("Could not find valid JSON in Mixtral response")
                    return self._fallback_scene_analysis_with_timing(prompt)
                    
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse Mixtral JSON: {str(e)}")
                logging.warning(f"Raw response: {result[:200]}...")
                return self._fallback_scene_analysis_with_timing(prompt)
            
        except Exception as e:
            logging.error(f"Error in Mixtral narrative analysis: {str(e)}")
            return self._fallback_scene_analysis_with_timing(prompt)
    
    def _fallback_scene_analysis_with_timing(self, prompt: str) -> List[Dict]:
        """Fallback scene analysis that returns timing data"""
        # Get basic scenes from existing fallback
        basic_scenes = self._fallback_scene_analysis(prompt)
        
        # Convert to timing format
        scenes_with_timing = []
        for i, scene in enumerate(basic_scenes):
            # Assign timing based on scene content
            duration = 3.0  # Default
            transition = "fade"
            
            # Adjust timing based on content
            if any(word in scene.lower() for word in ['standing', 'looking', 'posing']):
                duration = 2.5  # Shorter for static poses
            elif any(word in scene.lower() for word in ['walking', 'running', 'moving']):
                duration = 3.5  # Longer for movement
            elif any(word in scene.lower() for word in ['sitting', 'talking', 'examining']):
                duration = 4.0  # Longer for detailed actions
            
            # Adjust transitions
            if i == 0:
                transition = "none"  # First scene
            elif any(word in scene.lower() for word in ['suddenly', 'quickly', 'immediately']):
                transition = "cut"  # Quick transition
            elif any(word in scene.lower() for word in ['slowly', 'gently', 'gradually']):
                transition = "dissolve"  # Slow transition
            
            scenes_with_timing.append({
                'description': scene,
                'duration': duration,
                'transition': transition
            })
        
        logging.info(f"Fallback analysis generated {len(scenes_with_timing)} scenes with timing")
        return scenes_with_timing
    
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