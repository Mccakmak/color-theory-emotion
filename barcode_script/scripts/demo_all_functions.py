import sys
sys.path.append('../YT_CDB')
from utils.utils import *
from utils.barcode import *
from utils.transcription import *
from discover.discover import *
from topic_modelling.text_processing import *
from topic_modelling.LDA_modelling import *
from pprint import pprint
import pandas as pd
import glob
import pickle
from PIL import Image


def runAll(video_id):
    # enter a youtube video id
    video_id = 'cec3OpIpBio'

    # video_id = video_id
    timeNow = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    discovered_videos = discover_videos(video_id)
    save_video_to_csv(discovered_videos)

    # convert discovered videos to a dataframe

    discovered_videos_df = pd.DataFrame.from_dict(discovered_videos)
    
    # discovered_videos_df = pd.read_csv('videos_2023-02-13_095229.csv')
    discovered_video_ids = discovered_videos_df['id'].values
    # discover channels from channel_ids in discovered_videos_df

    subscriptions, featured_channels = discover_channels(discovered_videos_df['channelId'].unique())
    # save subscriptions and featured channels to pickle
    # with open(f'subscriptions_{timeNow}.pkl', 'wb') as f:
    #     pickle.dump(subscriptions, f)
    
    # with open(f'featured_channels_{timeNow}.pkl', 'wb') as f:
    #     pickle.dump(featured_channels, f)

    # load subscriptions and featured channels from pickle
    # with open('featured_channels_2023-02-13_100608.pkl', 'rb') as f:
    #     subscriptions = pickle.load(f)

    # with open('subscriptions_2023-02-13_100608.pkl', 'rb') as f:
    #     featured_channels = pickle.load(f)
    
    vid_to_chan = []

    for index, row in discovered_videos_df.iterrows():
        vid_to_chan.append({row['id']:[row['channelId']]})

    try:
    #build graph
        G = create_graph(video_id, np.unique(discovered_videos_df['id'].values), color='y')
        G = add_to_graph(G, vid_to_chan, color="b")
        G = add_to_graph(G, subscriptions, color="r")
        G = add_to_graph(G, featured_channels, color="g")

        print_graph(G, video_id)
    except Exception as e:
        print(e)
        #print error trace
        print(traceback.format_exc())
        print('error in creating graph')

    # scrpe comments and stats
    
    comments = []
    for video_id in discovered_video_ids:
        comments.append(get_comments(video_id))

    print(f'comments length: {len(comments)}')

    #save comments to pickle
    # with open(f'comments_{timeNow}.pkl', 'wb') as f:
    #     pickle.dump(comments, f)

    # # load comments from pickle
    # with open('comments_2023-01-27_065818.pkl', 'rb') as f:
    #     comments = pickle.load(f)

    flattened_comments = flatten_comments(comments)

    pd.DataFrame.from_dict(flattened_comments).to_csv(f'comments_{timeNow}.csv')
    # save comments to csv
    download_videos(discovered_video_ids)
    video_files = glob.glob('./videos/*.3gpp')
    print(video_files)
    # run videos to barcodes
    try:
        run_videos_to_barcodes(video_files)
        cluster_barcodes()
    except Exception as e:
        print(e)
        print('error in running videos to barcodes')

    # transcribe videos

    transcripts = get_transcripts(discovered_video_ids, timeNow)
    # transcripts = pd.read_csv('transcripts_2023-01-27_114539.csv')
   
    # topic model transcripts

    processed_transcripts = preprocess_transcripts(transcripts)

    extracted_topics = extract_topics(processed_transcripts)
    save_topics_to_csv(extracted_topics)


    