from discover.discover import discover_videos
from utils.utils import save_video_to_csv
import pandas as pd
from utils.transcription import get_transcripts


def main():
  videos = discover_videos('w2UV6HIpAog')
  # save videos to csv keeping all columns, name csv videos+timestamp
  save_video_to_csv(videos)

  # get video ids from videos
  video_ids = [video['id'] for video in videos]
  # get transcripts for videos
  transcripts = get_transcripts(video_ids)
  # save transcripts to csv labelling index as video_id
  pd.DataFrame.from_dict(transcripts, orient='index', columns=['transcript']).reset_index().rename(columns={'index':'video_id'}).to_csv(f'transcripts.csv')

if __name__ == "__main__":
  main()