from pytube import YouTube as yt
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm

# given a list of video ids, download videos using pytube
def download_videos(video_ids, filename):
  for video_id in tqdm(video_ids, total=len(video_ids), desc='Downloading videos'):
    download_video(video_id, filename)
#download videos
def download_video(video_id, filename):
  try:

    path = 'barcode_script/scripts/videos/'+ filename

    if not os.path.exists(path):
      os.makedirs(path)

    #add file name and extension
    yt('https://www.youtube.com/watch?v=' + video_id, use_oauth=True, allow_oauth_cache=True).streams.filter(res="144p").first().download(output_path=path, filename=video_id+'.3gpp')
  except Exception as e:
    print(e)

#function for saving video information as csv/to database
def save_video_to_csv(videos):
  time_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
  file_name = f'videos_{time_now}.csv'
  pd.DataFrame(videos).to_csv(file_name, index=False)
  return file_name

#def save topics to csv
def save_topics_to_csv(extracted_topics):
  formatted_topics = {}
  for video_id, topics in extracted_topics.items():
    #create a dictionary for each video
    video_topics = {}
    #for each topic in the list of topics
    for topic in topics:
      #add the topic number as key and topic probability as value
      try:
        video_topics[topic[0]] = topic[1]
      except Exception as e:
        video_topics = {'error': topics}
        break
    #add the video_topics dictionary to the extracted_topics dictionary
    formatted_topics[video_id] = video_topics

  #convert test to a dataframe
  formatted_topics = pd.DataFrame.from_dict(formatted_topics, orient='index')
  #save the dataframe to csv
  time_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
  formatted_topics.to_csv(f'extracted_topics_{time_now}.csv')

def flatten_comments(comments):
    flattened_comments = []
    for comment in comments:
        if comment is None:
            continue
        else:
          for c in comment:
              if c is None:
                  continue
              flattened_comments.append(c)
# make graph of channels

import networkx as nx
import matplotlib.pyplot as plt

def create_graph(parent, children, color):
  G = nx.DiGraph()
  G.add_node(parent, node_marker='circle', node_color='black')
  for child in children:
    if not G.has_node(child):
      G.add_node(child, node_marker='square', node_color=color)
    if not G.has_edge(parent, child):
      G.add_edge(parent, child, color=color)
  return G

#add nodes to graph
#edges has format [{parent:[children]}]
def add_to_graph(graph, edges, color):
  for edge in edges:
    for parent, children in edge.items():
      for child in children:
        if not graph.has_node(child):
          # add switch statement for node color
          switcher = {
              'b': 'diamond',
              'r': 'cross',
              'g': 'triangle'
          }
          shape = switcher.get(color)
          graph.add_node(child, node_marker=shape, node_color=color) 
        if not graph.has_edge(parent, child):
          graph.add_edge(parent, child, color=color)
  return graph

import matplotlib.patches as mpatches

def print_graph(G, name):
  plt.figure(figsize=(12, 10), dpi=80)
  edges = G.edges()
  colors = [G[u][v]['color'] for u,v in edges]

  # G.nodes return an array of tuples with the structure (node, {node_color: color, node_marker: marker})
  # extract the node_colors from tho nodes
  node_colors = [node[1]['node_color'] for node in G.nodes(data=True)]
  
  # print(node_colors)

  red_patch = mpatches.Patch(color='red', label='Channel Subscriptions discovery')
  blue_patch = mpatches.Patch(color='blue', label='Video to Channel discovery')
  green_patch = mpatches.Patch(color='green', label='Featured Channel Discovery')
  yellow_patch = mpatches.Patch(color='yellow', label='Video to Video discovery')

  plt.legend(handles=[ yellow_patch, blue_patch, red_patch, green_patch])
  pos = nx.kamada_kawai_layout(G)
  nx.draw(G, pos, edge_color='black',node_color=node_colors, with_labels=False)
  plt.savefig(name+".png", format="PNG")
  plt.show()

  nx.draw(G, pos, edge_color='black',node_color=node_colors, with_labels=True)
  plt.savefig(name+"_withLabel"+".png", format="PNG")
  nx.write_gexf(G, name+".gexf")
  # plt.show()
