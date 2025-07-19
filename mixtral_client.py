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

🎭 CRITICAL CHARACTER CONSISTENCY REQUIREMENTS:

CHARACTER PROFILE (MANDATORY - use EXACTLY as described):
{character_profile}

⚠️ ABSOLUTE CONSTRAINTS - NEVER VIOLATE:
1. START EVERY PROMPT with the EXACT character description from above
2. NEVER change clothing, hair, age, ethnicity, or physical features
3. NEVER add new clothing items or accessories not mentioned
4. NEVER change colors of existing clothing or hair
5. KEEP the same character throughout ALL 6 scenes of the story
6. ONLY change poses, expressions, and environmental context

✅ WHAT YOU CAN CHANGE:
- Character's pose and body position
- Facial expression and emotion
- Camera angle and composition
- Lighting and atmosphere
- Background environment
- Scene-specific actions

❌ WHAT YOU MUST NEVER CHANGE:
- Character's appearance, age, ethnicity
- Clothing items, colors, or style
- Hair color, length, or style
- Physical build or height
- Accessories or jewelry

CONTINUITY CHECK: Each scene should feel like the same person in different moments of the same story.

Respond in this JSON format:
{{
  "positive_prompt": "[EXACT CHARACTER DESCRIPTION], [scene-specific action/pose], [environment details], cinematic lighting, high quality, detailed",
  "negative_prompt": "multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person, different clothing, different hair, different appearance, different outfit, changed clothes, new clothes, different person, different character, different age, different ethnicity, inconsistent character"
}}

Make the positive prompt start with the exact character description, then add the scene-specific elements."""
            
            user_prompt = f"""Convert this scene into a detailed image generation prompt featuring the CONSISTENT CHARACTER:

🎭 MANDATORY CHARACTER PROFILE (copy EXACTLY):
{character_profile}

🎦 SCENE TO ADAPT: {scene}

📝 INSTRUCTIONS:
1. START your positive_prompt with the EXACT character description above (word for word)
2. Then add: ", [scene-specific action/pose from the scene description]"
3. Then add: ", [environmental details for this scene]"
4. End with: ", cinematic lighting, high quality, detailed, professional photography"
5. DO NOT modify any part of the character description
6. ONLY adapt the action/pose and environment to match the scene

EXAMPLE FORMAT:
"[EXACT CHARACTER DESCRIPTION], [doing scene action], [in scene environment], cinematic lighting, high quality, detailed"

⚠️ CRITICAL: The character must look identical across all scenes - same clothes, same hair, same everything except pose and expression.

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
        
        # Basic character based on scene content with VERY specific details for consistency
        if any(word in scenes_text.lower() for word in ['woman', 'she', 'her', 'girl', 'lady']):
            character = "25-year-old woman with shoulder-length straight brown hair, warm brown eyes, fair complexion, 5'6\" tall, slim build, wearing a bright red knee-length dress with short sleeves, white canvas sneakers with white laces, small silver stud earrings, friendly smile"
        elif any(word in scenes_text.lower() for word in ['man', 'he', 'his', 'guy', 'male']):
            character = "28-year-old man with short dark brown hair, blue eyes, medium complexion, 5'10\" tall, athletic build, wearing dark blue jeans, navy blue cotton t-shirt, white leather sneakers, confident expression"
        else:
            # Default to woman if unclear - same exact description
            character = "25-year-old woman with shoulder-length straight brown hair, warm brown eyes, fair complexion, 5'6\" tall, slim build, wearing a bright red knee-length dress with short sleeves, white canvas sneakers with white laces, small silver stud earrings, friendly smile"
        
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
        
        # Build prompt with character profile if available - character first for consistency
        if character_profile:
            positive_prompt = f"{character_profile}, {scene_single}, SAME EXACT CHARACTER, IDENTICAL APPEARANCE, SAME CLOTHING, consistent character throughout, cinematic lighting, high quality, detailed, professional photography, 4k resolution, dramatic composition, vivid colors"
        else:
            positive_prompt = f"single person, {scene_single}, one person only, consistent character, cinematic lighting, high quality, detailed, professional photography, 4k resolution, dramatic composition, vivid colors"
        
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
        negative_prompt = "multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person, several people, couple, pair, twins, family, team, friends, strangers, bystanders, different clothing, different hair, different appearance, changed outfit, different person, different character, new clothes, changed clothes, different dress, different shirt, different shoes, inconsistent character, character variation, different age, different ethnicity, wardrobe change"
        
        return {
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt
        }
    
    def analyze_narrative_structure(self, prompt: str) -> List[str]:
        """Use Mixtral to analyze narrative and extract optimal scene breaks"""
        
        system_prompt = """You are an expert story analyst and cinematographer. Your task is to create EXACTLY 6 sequential story frames that flow together coherently.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 6 frames (no more, no less)
- Each frame must logically follow the previous one
- Create a clear story progression from beginning to end
- Each frame should show different actions/poses but maintain story continuity
- Focus on creating a coherent narrative arc

For any prompt, create 6 sequential frames that tell a complete story:

Frame 1: Introduction/Setup (character introduction or scene setting)
Frame 2: Initial action/movement (character starts doing something)
Frame 3: Development/progression (action continues or develops)
Frame 4: Peak moment/interaction (main activity or key moment)
Frame 5: Resolution/ending action (activity concludes)
Frame 6: Final moment/conclusion (peaceful ending or final pose)

Example for "woman walking in park":
1. Woman standing at park entrance, looking ahead
2. Woman starting to walk along the path, gentle smile
3. Woman walking more confidently, enjoying the scenery
4. Woman stops to admire flowers, reaching toward them
5. Woman sitting on park bench, relaxing
6. Woman standing up from bench, looking content

CRITICAL: You must respond with ONLY valid JSON in this exact format:
{
  "scenes": [
    {
      "id": 1,
      "description": "Frame 1 description",
      "duration": 3.0,
      "transition": "fade"
    },
    {
      "id": 2,
      "description": "Frame 2 description",
      "duration": 3.0,
      "transition": "fade"
    },
    {
      "id": 3,
      "description": "Frame 3 description",
      "duration": 3.0,
      "transition": "fade"
    },
    {
      "id": 4,
      "description": "Frame 4 description",
      "duration": 3.0,
      "transition": "fade"
    },
    {
      "id": 5,
      "description": "Frame 5 description",
      "duration": 3.0,
      "transition": "fade"
    },
    {
      "id": 6,
      "description": "Frame 6 description",
      "duration": 3.0,
      "transition": "fade"
    }
  ]
}

Return ONLY the JSON, no other text."""
        
        user_prompt = f"""Create EXACTLY 6 sequential story frames for this prompt that flow together as a coherent narrative:

{prompt}

Create 6 frames that tell a complete story from beginning to end. Each frame should logically follow the previous one.

Return ONLY valid JSON with exactly 6 scenes in the format specified above."""
        
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
        """Fallback scene analysis when Mixtral is unavailable - creates exactly 6 story frames"""
        
        logging.info("Using fallback scene analysis to create 6 story frames")
        
        # Create exactly 6 coherent story frames based on the prompt
        base_prompt = prompt.lower()
        
        # Determine character and setting from prompt
        if any(word in base_prompt for word in ['woman', 'she', 'her', 'girl', 'lady']):
            character = "woman"
        elif any(word in base_prompt for word in ['man', 'he', 'his', 'guy', 'male']):
            character = "man"
        else:
            character = "person"
        
        # Determine main activity/setting
        if 'park' in base_prompt:
            location = "park"
            frames = self._create_park_story_frames(character)
        elif any(word in base_prompt for word in ['street', 'sidewalk', 'walking']):
            location = "street"
            frames = self._create_street_story_frames(character)
        elif any(word in base_prompt for word in ['mall', 'shopping', 'market']):
            location = "mall"
            frames = self._create_mall_story_frames(character)
        elif any(word in base_prompt for word in ['home', 'house', 'room']):
            location = "home"
            frames = self._create_home_story_frames(character)
        else:
            # Default story based on main activity mentioned
            frames = self._create_generic_story_frames(character, prompt)
        
        logging.info(f"Generated 6 story frames for {character} in {location if 'location' in locals() else 'generic setting'}")
        return frames
    
    def _create_park_story_frames(self, character: str) -> List[str]:
        """Create 6 coherent story frames for park setting"""
        return [
            f"{character} standing at the park entrance, looking ahead with anticipation",
            f"{character} starting to walk along the park path, taking in the scenery",
            f"{character} pausing to admire colorful flowers, reaching out gently",
            f"{character} sitting on a park bench, relaxing and enjoying the peaceful atmosphere",
            f"{character} feeding birds or watching children play, smiling warmly",
            f"{character} standing up from the bench, looking content and ready to leave"
        ]
    
    def _create_street_story_frames(self, character: str) -> List[str]:
        """Create 6 coherent story frames for street/sidewalk setting"""
        return [
            f"{character} standing at a street corner, checking directions",
            f"{character} beginning to walk down the sidewalk with purpose",
            f"{character} pausing to look at a shop window or street display",
            f"{character} crossing the street carefully, looking both ways",
            f"{character} continuing the walk, perhaps checking phone or watch",
            f"{character} arriving at destination, turning to face the camera with satisfaction"
        ]
    
    def _create_mall_story_frames(self, character: str) -> List[str]:
        """Create 6 coherent story frames for mall/shopping setting"""
        return [
            f"{character} entering the mall through the main entrance, looking around",
            f"{character} walking through the mall corridor, window shopping",
            f"{character} stopping at a store display, examining items with interest",
            f"{character} sitting in the food court area, taking a break",
            f"{character} carrying shopping bags, walking with satisfaction",
            f"{character} exiting the mall, looking pleased with the shopping experience"
        ]
    
    def _create_home_story_frames(self, character: str) -> List[str]:
        """Create 6 coherent story frames for home setting"""
        return [
            f"{character} entering through the front door, removing coat or shoes",
            f"{character} walking through the living room, settling in",
            f"{character} sitting on the couch or chair, getting comfortable",
            f"{character} preparing or enjoying a drink or snack",
            f"{character} reading, watching TV, or doing a relaxing activity",
            f"{character} standing up and stretching, looking refreshed"
        ]
    
    def _create_generic_story_frames(self, character: str, prompt: str) -> List[str]:
        """Create 6 coherent story frames for generic activities"""
        prompt_lower = prompt.lower()
        
        # Determine main activity from prompt
        if any(word in prompt_lower for word in ['walking', 'stroll', 'path']):
            activity = "walking"
            return [
                f"{character} standing ready to begin walking, looking ahead",
                f"{character} taking the first steps, starting the journey",
                f"{character} walking confidently, enjoying the movement",
                f"{character} pausing mid-walk to observe surroundings",
                f"{character} continuing the walk with renewed energy",
                f"{character} completing the walk, looking satisfied"
            ]
        elif any(word in prompt_lower for word in ['sitting', 'chair', 'bench']):
            activity = "sitting"
            return [
                f"{character} approaching the seating area, preparing to sit",
                f"{character} settling down into the seat, getting comfortable",
                f"{character} sitting relaxed, enjoying the moment",
                f"{character} adjusting position, finding the perfect comfort",
                f"{character} sitting peacefully, taking in the environment",
                f"{character} preparing to stand up, looking refreshed"
            ]
        elif any(word in prompt_lower for word in ['dancing', 'music', 'move']):
            activity = "dancing"
            return [
                f"{character} standing ready to dance, listening to the rhythm",
                f"{character} starting to move, feeling the music",
                f"{character} dancing gracefully, lost in the moment",
                f"{character} spinning or making an expressive gesture",
                f"{character} continuing to dance with joy",
                f"{character} finishing the dance with a final pose"
            ]
        else:
            # Default generic story progression
            return [
                f"{character} in the initial moment, looking ready and alert",
                f"{character} beginning the main activity, showing engagement",
                f"{character} fully involved in the activity, concentrated",
                f"{character} at the peak moment of the activity",
                f"{character} winding down the activity, showing satisfaction",
                f"{character} completing the activity, looking accomplished"
            ]
    
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