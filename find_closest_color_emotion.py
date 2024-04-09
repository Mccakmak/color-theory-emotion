import cv2
import webcolors
import pickle
import sys
import math
from collections import Counter
import copy
import os
import pandas as pd
from tqdm import tqdm
from barcode_script.scripts.download_and_barcode_videos import collect_barcode
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000

# Finds the lighter or darker versions of the given colors
def get_color_shades_and_tints(color_dict):
    # Create a barcode-like visualization for each color with more refined shades and tints

    colors = dict_color_to_rgb(color_dict)

    general_shades_tints = {color_name: generate_tints_and_shades_in_lab(color_name, rgb) for color_name, rgb in colors.items()}

    fig, axs = plt.subplots(len(colors), 1, figsize=(12, 20))

    for i, (color_name, shades_tints) in enumerate(general_shades_tints.items()):
        for j, shade_tint in enumerate(shades_tints):
            axs[i].fill_between([j, j + 1], 0, 1, color=np.array(shade_tint) / 255)
            axs[i].set_title(color_name)
            axs[i].axis("off")

    # Save the figure as a PDF
    file_path = "plots/color_shades_tints_graph.pdf"
    fig.savefig(file_path, format="pdf", bbox_inches="tight")

    return general_shades_tints

# Convert color strings to RGB values. It returns Color-RGB dictionary
def dict_color_to_rgb(color_dict):
    color_rgb_dict = {}
    for color in color_dict.keys():
        color_rgb_dict[color] = list(webcolors.name_to_rgb(color))
    return color_rgb_dict

"""
# Adjust the brightness
def adjust_rgb(color, percentage):
    adjusted = tuple(np.clip(int(c + (percentage / 100.0) * (255 if percentage > 0 else c)), 0, 255) for c in color)
    return adjusted
"""

def adjust_lab_lightness(lab_color, adjustment):
    """
    Adjust the lightness of a color in LAB color space.

    :param lab_color: The original color in LAB color space.
    :param adjustment: The amount to adjust the lightness by.
    :return: The adjusted color in LAB color space.
    """
    l, a, b = lab_color
    l_adjusted = max(min(l + adjustment, 100), 0)  # L* should be in the range [0, 100]
    return (l_adjusted, a, b)

def generate_tints_and_shades_in_lab(color_name, rgb_color, num_variations=10):
    """
    Generate tints and shades of a color in LAB color space.

    :param color_name: The name of the color.
    :param rgb_color: The original color in RGB color space (0-255 scale).
    :param num_variations: The number of tints and shades to generate.
    :return: A concatenated list of tints, the original color, and shades in RGB color space (0-255 scale).
    """
    # Normalize RGB and convert to LAB
    rgb_color_normalized = [x / 255.0 for x in rgb_color]
    lab_color = rgb2lab([[[rgb_color_normalized[0], rgb_color_normalized[1], rgb_color_normalized[2]]]])[0][0]

    # Define the lightness adjustments for tints and shades based on the color name
    if color_name == "white":
        lightness_tints = []
        lightness_shades = list(np.linspace(-1, -7, num_variations*2))  # darker
    elif color_name == "black":
        lightness_tints = list(np.linspace(5, 30, num_variations*2))  # lighter
        lightness_shades = []
    elif color_name == "gray":
        lightness_tints = list(np.linspace(10, 25, num_variations))  # lighter
        lightness_shades = list(np.linspace(-10, -25, num_variations))  # darker
    elif color_name == "yellow":
        lightness_tints = list(np.linspace(30, 60, num_variations))  # lighter
        lightness_shades = list(np.linspace(-10, -75, num_variations))  # darker
    elif color_name == "blue":
        lightness_tints = list(np.linspace(10, 25, num_variations))  # lighter
        lightness_shades = list(np.linspace(-3, -30, num_variations))  # darker
    elif color_name == "red":
        lightness_tints = list(np.linspace(5, 25, num_variations))  # lighter
        lightness_shades = list(np.linspace(-5, -55, num_variations))  # darker
    elif color_name == "green":
        lightness_tints = list(np.linspace(20, 50, num_variations))  # lighter
        lightness_shades = list(np.linspace(-10, -45, num_variations))  # darker
    elif color_name == "orange":
        lightness_tints = list(np.linspace(10, 25, num_variations))  # lighter
        lightness_shades = list(np.linspace(-10, -75, num_variations))  # darker
    elif color_name == "purple":
        lightness_tints = list(np.linspace(5, 25, num_variations))  # lighter
        lightness_shades = list(np.linspace(-5, -30, num_variations))  # darker
    elif color_name == "pink":
        lightness_tints = list(np.linspace(5, 15, num_variations))  # lighter
        lightness_shades = list(np.linspace(-5, -30, num_variations))  # darker
    # ... other colors ...
    else:
        # General case for other colors
        lightness_tints = list(np.linspace(20, 50, num_variations))  # lighter
        lightness_shades = list(np.linspace(-20, -50, num_variations))  # darker

    # Generate tints (lighter versions)
    tints_lab = [adjust_lab_lightness(lab_color, adjustment) for adjustment in lightness_tints]
    tints_rgb = [lab2rgb([[color]])[0][0] for color in tints_lab]
    tints_rgb = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in tints_rgb]  # convert back to 0-255 scale

    # Generate shades (darker versions)
    shades_lab = [adjust_lab_lightness(lab_color, adjustment) for adjustment in lightness_shades]
    shades_rgb = [lab2rgb([[color]])[0][0] for color in shades_lab]
    shades_rgb = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in shades_rgb]  # convert back to 0-255 scale

    return tints_rgb[::-1] + [rgb_color] + shades_rgb  # original color is in RGB

"""
# Specific adjustments for each color for shades and tints. To generate better distribution.
def general_adjustments(color_name, color):
    if color_name == "white":
        tints = []
        shades = [adjust_rgb(color, -i) for i in np.linspace(1, 7, 20)]  # 2.5% to 17.5% shades
    elif color_name == "black":
        tints = [adjust_rgb(color, i) for i in np.linspace(5, 30, 20)]  # 2.5% to 17.5% tints
        shades = []
    elif color_name == "gray":
        tints = [adjust_rgb(color, i) for i in np.linspace(10, 25, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(10, 35, 10)]  # 30% to 60% shades
    elif color_name == "pink":
        tints = [adjust_rgb(color, i) for i in np.linspace(5, 20, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(25, 50, 10)]  # 30% to 60% shades
    elif color_name == "yellow":
        tints = [adjust_rgb(color, i) for i in np.linspace(20, 60, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(20, 50, 10)]  # 30% to 60% shades
    elif color_name == "blue":
        tints = [adjust_rgb(color, i) for i in np.linspace(10, 50, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(20, 60, 10)]  # 30% to 60% shades
    elif color_name == "red":
        tints = [adjust_rgb(color, i) for i in np.linspace(20, 60, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(20, 60, 10)]  # 30% to 60% shades
    elif color_name == "orange":
        tints = [adjust_rgb(color, i) for i in np.linspace(10, 40, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(20, 60, 10)]  # 30% to 60% shades
    else:
        # General case for other colors
        tints = [adjust_rgb(color, i) for i in np.linspace(20, 50, 10)]  # 30% to 60% tints
        shades = [adjust_rgb(color, -i) for i in np.linspace(20, 50, 10)]  # 30% to 60% shades

    return tints[::-1] + [color] + shades
"""




# Find the frame per second number for a video.
def find_fps(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    #frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return fps

# Get sub-part of the barcodes.
def get_chunks(barcode, sec, fps):
    
    img = cv2.imread(barcode)
    
    # RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Take vertical bars 
    img = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    chunk_count = round(sec*fps)
    bar_count = len(img)
    
    bar_chunks = []
    i = 0
    # Loop through first bars end is not included
    for x in range(int(bar_count/chunk_count)):
        bar_chunks.append(img[i:i+chunk_count])
        i = i+chunk_count
    # Last one may not be equal to chunk_count
    bar_chunks.append(img[i:])
    
    return bar_chunks

"""
#Euclidian Distance for RGB-RGB
def calculate_color_distance_euclidian(rgb_color, shades_color):
    distances = [math.sqrt((rgb_color[0] - shade_color[0]) ** 2 + (rgb_color[1] - shade_color[1]) ** 2 + (
                rgb_color[2] - shade_color[2]) ** 2) for shade_color in shades_color]
    return min(distances), sum(distances) / len(distances)
"""

def calculate_color_distance(rgb_color1, shade_color):
    """
    Calculate the CIEDE2000 color difference between two colors.

    :param rgb_color1: The first color as an RGB tuple.
    :param shade_color: The second color as an RGB tuple.
    :return: The CIEDE2000 color difference between the two colors.
    """
    # Normalize the RGB colors if they are on a 0-255 scale
    rgb_color1 = [x / 255.0 for x in rgb_color1]
    shade_color = [x / 255.0 for x in shade_color]

    # Convert the RGB colors to LAB colors
    lab_color1 = rgb2lab([[[rgb_color1[0], rgb_color1[1], rgb_color1[2]]]])[0][0]
    lab_color2 = rgb2lab([[[shade_color[0], shade_color[1], shade_color[2]]]])[0][0]

    # Calculate the CIEDE2000 color difference
    return deltaE_ciede2000(lab_color1, lab_color2)


# Calculate all the color distances. Wrapper of calculate distance
def calculate_color_distances(rgb_color, shades_color):
    distances = []
    for shade_color in shades_color:
        distances.append(calculate_color_distance(rgb_color, shade_color))

    # Return minimum distance and the average distance
    return min(distances), sum(distances)/len(distances)

# Finds the closest color for a given RGB value
def find_closest_color(chunks, color_shades_rgb_dict):
    all_closest_colors = []

    # For each chunk(stacked bars)
    for chunk in chunks:
        closest_colors_chunk = []
        # For each bar
        for bar in chunk:
            min_distance = sys.maxsize
            avg_distances = {}
            closest_colors = []

            # Take the RGB color from the bar
            rgb_color = bar[0]

            # For each color in color dictionary
            for color, shades in color_shades_rgb_dict.items():
                # Find the closest color by checking each shade for each color
                min_dist, avg_dist = calculate_color_distances(rgb_color, shades)

                if min_dist < min_distance:
                    closest_colors = [color]
                    min_distance = min_dist
                elif min_dist == min_distance:
                    closest_colors.append(color)

                avg_distances[color] = avg_dist

            # If more than one color has the same minimum distance, find the color with the lowest average distance
            if len(closest_colors) > 1:
                closest_color = min(closest_colors, key=lambda x: avg_distances[x])
            else:
                closest_color = closest_colors[0]
            closest_colors_chunk.append(closest_color)
        all_closest_colors.append(closest_colors_chunk)
    return all_closest_colors


# What is the color percentages overall?
def find_color_percentage(color_chunks):
    color_counts = {}

    for color_chunk in color_chunks:
        for color in color_chunk:

            # If the chunks contains RGB values
            if type(color[0][0]) == np.uint8:
                # Take the RGB color from the bar
                rgb_color = color[0]

                # Find color name using RGB values
                color = find_closest_color_name(rgb_color)

            if color in color_counts:
                # If we have the color increase 1
                color_counts[color] += 1
            else:
                # If we dont have the color initialize as 1
                color_counts[color] = 1

    # Find the sum of all counts
    sum_counts = sum(color_counts.values())

    # Normalize the counts between 0 and 1, with sum of values equal to 1
    normalized_counts = {}

    for color, count in color_counts.items():
        normalized_counts[color] = count / sum_counts

    # Sort and return
    return sort_dict(normalized_counts)


# RGB to Color Name
def find_closest_color_name(rgb_val):
    min_colors = {}
    for key, name in webcolors.CSS3_NAMES_TO_HEX.items():
        # Convert the color name to RGB values
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)

        # Calculate the squared differences between RGB values
        rd = (r_c - rgb_val[0]) ** 2
        gd = (g_c - rgb_val[1]) ** 2
        bd = (b_c - rgb_val[2]) ** 2

        # Store the color name and the corresponding squared difference as a key-value pair
        min_colors[(rd + gd + bd)] = key

    # Retrieve the color name with the minimum squared difference
    closest_color_name = min_colors[min(min_colors.keys())]

    return closest_color_name


# Find emotion values using the Dictionary we have by giving closest RGB values
def color_to_emotion(closest_colors, color2emotion_weight_dict):
    
    all_chunk_emotion_dicts = []

    for chunk in closest_colors:

        chunk_color2emotion_dict = copy.deepcopy(color2emotion_weight_dict)
        chunk_size = len(chunk)

        # How many colors are there in one chunk
        chunk_colors = dict(Counter(chunk))

        for color, emotion_dict in chunk_color2emotion_dict.items():
            if color in chunk_colors:
                emotion_dict.update((emotion, prob*chunk_colors[color]/chunk_size) for emotion, prob in emotion_dict.items())
            else:
                emotion_dict.update((emotion, prob*0) for emotion, prob in emotion_dict.items())

        all_chunk_emotion_dicts.append(chunk_color2emotion_dict)
    
    return all_chunk_emotion_dicts


# It finds the overall emotion for one video.
def find_overall_emotion(chunk_emotions):
    # Take one example and find the unique emotions
    unique_emotions = find_unique_emotions(chunk_emotions[0])

    # Find chunk length
    chunk_length = len(chunk_emotions)

    # Count the emotion weights and sum
    chunk_emotion_values_dict = {}

    # Initialize the dictionary with unique emotions with 0 values
    for emotion in unique_emotions:
        chunk_emotion_values_dict[emotion] = 0

    for color_emotion_dict in chunk_emotions:
        for emotion_and_weight in color_emotion_dict.values():
            for emotion, weight in emotion_and_weight.items():
                # Sum each weight
                chunk_emotion_values_dict[emotion] += weight

    normalized_chunk_emotion_values_dict = normalize_dict(chunk_emotion_values_dict, chunk_length)
    sorted_normalized_chunk_emotion_values_dict = sort_dict(normalized_chunk_emotion_values_dict)

    return sorted_normalized_chunk_emotion_values_dict


# Find the unique emotions inside the 2D dictionary.
def find_unique_emotions(chunk_emotions):
    # Initialize an empty set
    unique_emotions = set()  
    # Iterate over each sub-dictionary
    for emotion_and_weight in chunk_emotions.values():
        # Iterate over each value in the sub-dictionary and add it to the set
        for emotion in emotion_and_weight.keys():
            unique_emotions.add(emotion)
    return unique_emotions

# Normalize the values between 0-1 in the dictionary
def normalize_dict(dictionary, const):
    
    # Normalization
    for emotion, weight in dictionary.items():
        dictionary[emotion] = weight/const
    return dictionary

def sort_dict(dictionary):
    
    # Sort the dictionary by value in descending order
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_dict


def process_video(args):
    video, barcode, color_shades_rgb_dict, color2emotion_weight_dict, video_path, barcode_path, seconds_per_chunk = args

    fps = find_fps(video_path + video)
    chunks = get_chunks(barcode_path + barcode, seconds_per_chunk, fps)
    closest_colors = find_closest_color(chunks, color_shades_rgb_dict)

    # Retrieve the overall color distribution
    color_percentages_unmapped = find_color_percentage(chunks)
    color_percentages_mapped = find_color_percentage(closest_colors)

    # Retrieve emotions
    chunk_emotions = color_to_emotion(closest_colors, color2emotion_weight_dict)
    emotion_percentages = find_overall_emotion(chunk_emotions)

    # Extract video_id
    video_id = video.split('.')[0]

    return video_id, emotion_percentages, color_percentages_unmapped, color_percentages_mapped

def get_emotion_words(color_shades_rgb_dict, color2emotion_weight_dict, filename, per_chunk_sec=5):
    video_path = 'barcode_script/scripts/videos/' + filename + '/'
    barcode_path = 'barcode_script/scripts/barcode/' + filename + '/'

    video_list = os.listdir(video_path)
    barcode_list = os.listdir(barcode_path)

    # Chunks are divided into x seconds
    seconds_per_chunk = per_chunk_sec

    emotion_words_dict = {}

    # Prepare the list of arguments for each call of process_video
    args_list = [
        (video, barcode, color_shades_rgb_dict, color2emotion_weight_dict, video_path, barcode_path, seconds_per_chunk)
        for video, barcode in zip(video_list, barcode_list)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_video, args) for args in args_list]

        kwargs = {
            'total': len(futures),
            'unit': 'video',
            'unit_scale': True,
            'leave': True,
            'desc': 'Mapping colors to emotions:'
        }

        for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
            try:
                video_id, emotion_percentages, color_percentages_unmapped, color_percentages_mapped = f.result()
                emotion_words_dict[video_id] = emotion_percentages
            except ZeroDivisionError:
                print(f"Skipped {f} due to division by zero.")
            except Exception as e:
                print(f"An exception occurred: {e}")

    return emotion_words_dict

def retrieve_emotion_words_from_video(filename):
    
    # Read Color Emotion Table that we have collected from Literature Reviews
    with open('color2emotion_weight_dict.pkl', 'rb') as f:
        color2emotion_weight_dict = pickle.load(f)

    # Color shades
    color_shades_rgb_dict = get_color_shades_and_tints(color2emotion_weight_dict)

    # Generate barcodes
    df = pd.read_csv('inputs/'+ filename + '.csv')
    collect_barcode(df.video_id, filename)

    # Emotion Words
    emotion_words_dict = get_emotion_words(color_shades_rgb_dict, color2emotion_weight_dict, filename, per_chunk_sec = 100)

    return emotion_words_dict

if __name__ == '__main__':
    retrieve_emotion_words_from_video("asonam_uyghur_video_ids")
