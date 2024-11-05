import json





# source_path='/nesa_data/remote_shome/chensn/workspace/LLaMA-Factory/caption/results/v1/caption_output/caption/results.json'

'''
from this form

{
        "id": "video7e2f6d3ac77221c67690175325cba767-scene12",
        "messages": [
            {
                "role": "system",
                "content": "You are an excellent video captioner."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12.mp4",
                        "max_pixels": 151200,
                        "fps": 1.0
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Before generating the video description, please first identify and list all entity names present in the video. Then, provide a detailed description of each entity based on their order in the list.  The description should include, but is not limited to, the object's shape, color, appearance, location, size, orientation, and motion.\n\n**Entity Identification and Description:**\n\n**Entity List:** [Entity 1], [Entity 2], ... [Entity n] \n\n**Entity Descriptions:**\n\n1. **[Entity 1]:** \n    - Description: [Detailed description of entity 1]\n2. **[Entity 2]:**\n    - Description: [Detailed description of entity 2]\n  …\nn. **[Entity n]:**\n    - Description: [Detailed description of entity n] \n\nOnce you have identified and described all entities, please generate a structured video description based on the following content. The description should include five parts: overall scene description, background scene description, camera position and movement description, additional detail description, and keywords. Please ensure the description is detailed, accurate, and covers all important information in the video.\n\n**Video Description Generation:**\n\n1. **Overall Scene Description**:\n   - Describe the appearance, location, and actions of the main objects in the scene as detail as possible.\n   - Describe how the main objects change over time in detail.\n\n2. **Background Scene Description**:\n   - Describe the background of the scene, including lighting, colors, and style in detail.\n\n3. **Camera Position and Movement Description**:\n   - Describe the camera's perspective and position in detail.\n   - Describe how the camera position changes over time in detail.\n\n4. **Additional Detail Description**:\n   - Describe other objects besides the main ones and their locations in detail.\n   - Describe the relationship and changes over time between these objects and the main ones in detail.\n\n5. **Keywords**:\n   - Including but not limited to the following:\n     a. **Camera Style**: Such as low angle, FPV, Hand held, wide angle, close up, tracking, 50mm lens, etc.\n     b. **Lighting Style**: Such as diffused lighting, lens flare, side lit, gel lighting, venetian lighting, etc.\n     c. **Motion Speed**: Such as dynamic motion, slow motion, fast motion, timelapse, etc.\n     d. **Motion Type**: Such as grows, emerges, explodes, wraps, transforms, shatters, unfolds, etc.\n     e. **Aesthetic Style**: Such as cinematic, iridescent, home video VHS, glitchcore, etc.\n     f. **Text Style (if there is text)**: Such as bold, graffiti, neon, embroidery, etc.\n\nPlease fill in the description in the following format:\n\n---\n\n**Entity Identification and Description:**\n**Entity List:** [Entity 1], [Entity 2], ... [Entity n] \n\n**Entity Descriptions:**\n1. **[Entity 1]:** \n    - Description: [Detailed description of entity 1]\n…\n\n**Overall Scene Description**:\n[Description content]\n\n**Background Scene Description**:\n[Description content]\n\n**Camera Position and Movement Description**:\n[Description content]\n\n**Additional Detail Description**:\n[Description content]\n\n**Keywords**:\n- Camera Style: [Keyword]\n- Lighting Style: [Keyword]\n- Motion Speed: [Keyword]\n- Motion Type: [Keyword]\n- Aesthetic Style: [Keyword]\n- Text Style (if there is text): [Keyword]\n\n---\n\nPlease fill in the video description according to the above format.\n\nImportant Instructions:\n1. When generating the video description, use language that resembles instructions for video creation rather than third-person observations.\n2. Avoid phrases like 'The video shows' or 'We can see'. Instead, use direct, instructional language.\n3. Use present tense and active voice to describe actions and scenes.\n4. Frame the description as if you're providing instructions to create the video, not describing an existing video.\n5. Be specific and detailed in your instructions, as if guiding a video creator or AI to generate the exact scene.\n\nFor example, instead of 'The video shows a red car driving down a street', write 'A vibrant red car speeds down a bustling urban street, its sleek body gleaming under the city lights.'\n\n\n"
                    }
                ]
            }
        ],
        "video-path": "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12.mp4",
        "raw_img_list": [
            "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12_frame_0.png",
            "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12_frame_12.png",
            "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12_frame_24.png",
            "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12_frame_37.png",
            "/remote_data/backup/a100-backup/a100-2-data/youtube_test/youtube_video_clips/7e2f6d3ac77221c67690175325cba767/video7e2f6d3ac77221c67690175325cba767_scene12_frame_49.png"
        ],
        "time_stamp": [
            0.0,
            0.31,
            0.62,
            0.96,
            1.27
        ],
        "save_key": "caption",
        "caption": "**Entity Identification and Description:**\n**Entity List:** [Chimpanzee]\n\n**Entity Descriptions:**\n1. **[Chimpanzee]:** \n    - Description: A chimpanzee with a dark brown fur coat, large expressive eyes, and a curious expression on its face. The chimpanzee appears to be looking upwards, possibly observing something in the sky or listening to sounds from above.\n\n**Overall Scene Description**:\nThe scene features a chimpanzee in a natural forest setting. The background is filled with lush green foliage, creating a dense and verdant environment. The lighting is soft and natural, suggesting it might be daytime. The chimpanzee is the central focus of the scene, positioned slightly off-center to the left.\n\n**Background Scene Description**:\nThe background consists of dense green leaves and branches, indicating a tropical or subtropical forest habitat. The light filtering through the canopy creates a dappled effect on the ground, adding depth and texture to the scene.\n\n**Camera Position and Movement Description**:\nThe camera is positioned at eye level with the chimpanzee, allowing for a clear view of its face and upper body. The camera remains stationary throughout the scene, focusing on the chimpanzee's expressions and movements.\n\n**Additional Detail Description**:\nThe chimpanzee's fur is thick and glossy, reflecting the surrounding light. Its ears are small and rounded, blending seamlessly into its head. The chimpanzee's nose is prominent and slightly curved, adding to its expressive face. The chimpanzee's mouth is closed, and its lips are slightly parted, giving it a contemplative look.\n\n**Keywords**:\n- Camera Style: Eye level\n- Lighting Style: Natural, dappled\n- Motion Speed: Still\n- Motion Type: None\n- Aesthetic Style: Natural, serene\n- Text Style (if there is text): None",
        "gen_model_id": "qwen-2-vl"
    },



'''


# goal_path='./data/convert_mul_pic_241101.py'

'''
change to this form


    {
        "image": [
            "/wangbenyou/guimingchen/datasets/llava/svit/vg/VG_100K/2364176.jpg",
            "/wangbenyou/guimingchen/datasets/llava/svit/ocr_vqa/images/1515079341.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<img><image></img>\nPlease provide a short description for this region: [0.06, 0.46, 0.33, 0.78]."
            },
            {
                "from": "gpt",
'''
import json
import random
import tqdm

source_path = '/nesa_data/remote_shome/chensn/workspace/LLaMA-Factory/caption/results/v1/caption_output/caption/results.json'
goal_path = './data/convert_mul_pic_241101.json'

simple_prompts = [
    "Describe this image in detail, including the main subject and background.",
    "Provide a comprehensive description of what you see in this image.",
    "Explain the scene depicted in this image, including all important elements.",
    "Give a detailed overview of the contents of this image.",
    "Describe the main subject, its surroundings, and any notable features in this image.",
    "Provide a thorough description of the visual elements in this image.",
    "Explain what's happening in this image, including the setting and main subject.",
    "Describe the overall scene and key details visible in this image.",
    "Give a detailed account of the subject and its environment in this image.",
    "Provide a comprehensive explanation of the visual content in this image."
]
# 10

medium_prompts = [
    "Describe this image in detail, including the main subject, background, and any notable features. Also, comment on the lighting and overall mood.",
    "Provide a comprehensive description of the image, including the subject, setting, and any interactions or movements. Discuss the composition and visual elements.",
    "Analyze the contents of this image, describing the main elements, their characteristics, and how they relate to each other. Include observations about the scene's atmosphere.",
    "Give a detailed overview of the image, including the primary subject, its surroundings, and any significant details. Comment on the visual style and any implied narrative.",
    "Describe the scene depicted in this image, focusing on the main subject, background elements, and any notable actions or expressions. Include observations about lighting and color.",
    "Provide a thorough description of the image, including the central focus, supporting elements, and overall composition. Discuss any interesting visual aspects or implied meanings.",
    "Explain the visual content of this image in detail, including the main subject, setting, and any secondary elements. Comment on the mood and any notable artistic choices.",
    "Offer a comprehensive analysis of the image, describing the primary subject, its environment, and any significant details. Include observations about the visual style and atmosphere.",
    "Describe the image's content in depth, including the main focus, background elements, and any notable features. Discuss the overall composition and any implied narrative or emotions.",
    "Provide a detailed explanation of the image, including the central subject, surrounding elements, and any interesting visual aspects. Comment on the mood and any stylistic choices."
]
# 10

complex_prompts = [
    "Provide a detailed analysis of this image, including its composition, subject matter, use of color and light, and any symbolic or thematic elements. Consider the image's potential context and meaning. Describe the main subject in depth, including its appearance, actions, and relationship to its surroundings. Analyze the background and any secondary elements, explaining how they contribute to the overall scene. Comment on the camera perspective and any notable visual techniques used.",
    "Conduct a comprehensive visual analysis of this image. Begin by identifying and describing the main subject in detail, including its physical characteristics, position, and any actions or expressions. Then, examine the background and setting, noting colors, textures, and atmospheric elements. Discuss the composition, including the use of space, balance, and visual hierarchy. Analyze the lighting, its sources, and effects on the scene. Consider any implied narrative or emotional content, and how the various elements contribute to this. Finally, reflect on the overall style and any potential symbolic or thematic significance.",
    "Offer an in-depth interpretation of this image, addressing its literal content, compositional structure, use of visual elements, and potential symbolic meanings. Start by describing the main subject in great detail, including its appearance, behavior, and relationship to its environment. Analyze the background, noting its components and how they frame or contextualize the main subject. Examine the use of color, light, and shadow throughout the image. Discuss the camera angle and perspective, and how these choices affect the viewer's perception. Consider the mood or atmosphere evoked by the image, and any narrative elements suggested. Conclude with thoughts on the overall visual impact and any broader themes or messages conveyed.",
    "Provide an exhaustive description and analysis of this image. Begin with a detailed account of the main subject, including its physical attributes, positioning, and any actions or expressions. Describe the setting in depth, including all visible elements and their spatial relationships. Analyze the composition, discussing how various elements are arranged and how this guides the viewer's eye. Examine the use of color, contrast, and texture throughout the image. Discuss the lighting, its quality, direction, and effects on the scene. Consider the mood or atmosphere created and how different elements contribute to this. Reflect on any implied narrative or symbolic content. Finally, offer insights into the overall visual style and any technical or artistic techniques employed.",
    "Present a thorough visual analysis of this image, covering all aspects from content to style. Start by identifying and describing the main subject in exacting detail, noting its appearance, posture, and any actions or expressions. Examine the background and setting comprehensively, describing all visible elements and their contribution to the scene. Analyze the composition, including the use of lines, shapes, and space. Discuss the color palette and its effects on the mood and visual impact. Describe the lighting, its sources, quality, and how it affects the scene. Consider the perspective and framing, and how these choices influence the viewer's perception. Reflect on any narrative elements or emotional content suggested by the image. Conclude with observations on the overall style, any notable artistic techniques, and potential symbolic or thematic significance."
]
# 5

prompt_pool = simple_prompts + medium_prompts + complex_prompts
# 2: 2: 1

with open(source_path, 'r') as file:
    source_data = json.load(file)

converted_data = []
one_image_placeholder='<img><image></img>\n'

# debug_limit=10

for item in tqdm.tqdm(source_data):
# for item in tqdm.tqdm(source_data[:debug_limit]):
    images = item['raw_img_list']
    caption = item['caption']

    random_prompt = random.choice(prompt_pool)

    new_item = {
        "image": images,
        "conversations": [
            {
                "from": "human",
                "value": f"{one_image_placeholder*len(images)}{random_prompt}"
            },
            {
                "from": "gpt",
                "value": caption
            }
        ]
    }

    converted_data.append(new_item)

# 写入转换后的数据
with open(goal_path, 'w') as file:
    json.dump(converted_data, file, indent=4)

print(f"Conversion complete. Output saved to {goal_path}")

