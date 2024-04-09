from collections import Counter
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np

# Emotion Counts for each color
def evaluate_weights(color2emotion_weight_dic, color2emotion_dic):
    for color in color2emotion_dic.keys():

        # Ignore mixed colors for now
        if '-' in color:
            continue

        # Find the count of the emotion in the emotion list and save it in the dictionary
        color2emotion_weight_dic[color] = dict(
            Counter([item for sublist in color2emotion_dic[color] for item in sublist]))

    return color2emotion_weight_dic


# Add only half weight
def add_half_weight(color2emotion_weight_dic, emotion, count, color):
    # If the emotion is there, add half of the count value
    if emotion in color2emotion_weight_dic[color]:
        color2emotion_weight_dic[color][emotion] += (count / 2)
    # If the emotion is new, put the half of the count value
    else:
        color2emotion_weight_dic[color][emotion] = (count / 2)


# Mixed_emotions and their counts.
def find_attr_mixed_colors(color2emotion_dic, color):
    mix_color_emotions = dict(Counter([item for sublist in color2emotion_dic[color] for item in sublist])).keys()
    mix_color_counts = dict(Counter([item for sublist in color2emotion_dic[color] for item in sublist])).values()

    return mix_color_emotions, mix_color_counts

# For mixed colors add the occurrences.
def include_mixed_colors(color2emotion_weight_dic, color2emotion_dic):
    for color in color2emotion_dic.keys():
        if '-' in color:
            # find mixed colors
            color1, color2 = color.split('-')

            mix_color_emotions, mix_color_counts = find_attr_mixed_colors(color2emotion_dic, color)

            # Adding half weight
            for emotion, count in zip(mix_color_emotions, mix_color_counts):
                add_half_weight(color2emotion_weight_dic, emotion, count, color1)
                add_half_weight(color2emotion_weight_dic, emotion, count, color2)

    return color2emotion_weight_dic


def make_percentage(color2emotion_weight_dic, color):
    total = sum(color2emotion_weight_dic[color].values())
    for key, val in color2emotion_weight_dic[color].items():
        color2emotion_weight_dic[color][key] = format(val / total, '.4f')
    return color2emotion_weight_dic


def find_weights(color2emotion_weight_dic):

    # Make a copy so that the value of the original dictionary will not change
    color2emotion_weight_dic_copy = copy.deepcopy(color2emotion_weight_dic)

    for color in color2emotion_weight_dic_copy.keys():
        color2emotion_weight_dic_copy = make_percentage(color2emotion_weight_dic_copy, color)
    return color2emotion_weight_dic_copy


def plot_data(data, colors_to_plot, file_name):
    # Prepare data for plotting
    xtick_positions = []
    xtick_labels = []

    current_position = 0
    for color in colors_to_plot:
        for emotion in data[color]:
            xtick_positions.append(current_position)
            xtick_labels.append(emotion)
            current_position += 1
        current_position += 1  # Add a space between color groups

    # Create a figure and axis with adjusted font size
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(22, 6))

    # Create bars with a single color for each emotion within a color
    current_position = 0
    for color in colors_to_plot:
        for emotion in data[color]:
            # If the color is white put border around it
            if color == "white":
                ax.bar(current_position, float(data[color][emotion]), color=color, edgecolor="black", linewidth=0.5)
            else:
                ax.bar(current_position, float(data[color][emotion]), color=color)
            current_position += 1
        current_position += 1  # Add a space between color groups

    # Set labels, title and adjust margins
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=90, ha='center')
    ax.set_ylabel('Emotion Weight')
    ax.set_xlabel('Emotion')
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.margins(x=0.005)  # Reduce margins to minimize unused space

    # Save the figure as a PDF
    fig.savefig(file_name, format="pdf", bbox_inches="tight")

def main():
    """
    -Excitement(Eagerness, Enthusiasm, Elation, Energetic, Powerful, Determination, Confident)

    -Happiness(Satisfaction, Pride, Amusement, Admiration, Joy)

    -Love(Passion)

    -Hope(Faith)

    -Gratitude

    -Peaceful(Serenity, Relief, Calm)

    -Sadness(Sorrow, Despair, Disappointment, Discomfort)

    -Shame(Embarrassment, Guilt, Regret, Remorse)

    -Surprise(Shock, Disbelief, Awe, Confusion)

    -Pity(Sympathy)

    -Fear(Anxiety, Terror, Horror, Panic, Apprehension)

    -Anger(Annoyance, Frustration, Irritation, Rage)

    -Jealousy(Envy, Greed)

    -Disgust(Revulsion, Contempt, Disrespect)

    -Boredom(Indifference)

    -Curiosity(interest)
    """

    # Color and Emotion Mapping from Literature Reviews:
    color2emotion_dic = {
        'yellow': [['joy'], ['fear', 'happiness', 'joy'], ['happiness'], ['joy', 'happiness'], ['fear'],
                   ['ecstasy', 'joy', 'serenity'], ['happiness']],
        'yellow-red': [['powerful'], ['energetic', 'excited']],
        'red-yellow': [['happiness']],
        'yellow-orange': [['optimism']],
        'blue': [['sadness'], ['confident', 'sadness'], ['calm'], ['trust'], ['confusion'],
                 ['grief', 'sadness', 'pensive', 'amazement', 'surprise', 'distraction', 'disapproval'], ['calm']],
        'red-blue': [['discomfort']],
        'blue-red': [['discomfort']],
        'red': [['faith'], ['anger', 'love'], ['anger', 'love'], ['powerful', 'anger'], ['anger'],
                ['rage', 'anger', 'annoyance'], ['anger']],
        'green': [['calm'], ['faith', 'greed'], ['comfort', 'hopeful', 'peaceful'], ['greed'], ['greed'],
                  ['admiration', 'trust', 'acceptance', 'terror', 'fear', 'apprehension', 'submission'],
                  ['disgust', 'envious', 'jealousy']],
        'orange': [['joy', 'determination'], ['joy', 'happiness'], ['shame'],
                   ['vigilance', 'anticipation', 'interest']],
        'orange-red': [['aggressive']],
        'purple': [['introspective', 'melancholic'], ['tired'], ['sadness'], ['powerful'],
                   ['loathing', 'disgust', 'boredom']],
        'green-yellow': [['disgust', 'annoyance'], ['love']],
        'blue-green': [['annoyance', 'confusion', 'sick'], ['awe']],
        'purple-blue': [['calm', 'powerful'], ['sadness']],
        'red-purple': [['love'], ['contempt']],
        'white': [['innocence', 'lonely', 'peaceful'], ['calm']],
        'gray': [['boredom', 'confusion', 'depressive', 'sadness']],
        'black': [['powerful'], ['depressive', 'powerful', 'fear'], ['fear']],
        'pink': [['surprise']]
        }

    color2emotion_weight_dic = {}


    evaluate_weights(color2emotion_weight_dic, color2emotion_dic)
    print("Emotion Counts without Mixed Colors:")
    print(color2emotion_weight_dic)
    print("Normalized Emotion Distribution without Mixed Colors:")
    print(find_weights(color2emotion_weight_dic))

    include_mixed_colors(color2emotion_weight_dic, color2emotion_dic)
    print("Emotion Counts:")
    print(color2emotion_weight_dic)
    print("Normalized Emotion Distribution:")
    final_weights = find_weights(color2emotion_weight_dic)
    print(final_weights)
    # Define the color groups for the two PDFs
    first_group_colors = list(final_weights.keys())[:4]  # First three colors
    second_group_colors = list(final_weights.keys())[4:10]  # Next seven colors

    # Plot and save the first group
    plot_data(final_weights, first_group_colors, "plots/color_emotion_dictionary_graph_part1.pdf")

    # Plot and save the second group
    plot_data(final_weights, second_group_colors, "plots/color_emotion_dictionary_graph_part2.pdf")

    # Save the Dictionary for future usage
    with open('color2emotion_weight_dict.pkl', 'wb') as f:
        for key, emotion_dict in final_weights.items():
            emotion_dict.update((key, float(val)) for key, val in emotion_dict.items())
        pickle.dump(final_weights, f)


if __name__ == '__main__':
    main()





